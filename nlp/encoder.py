import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from schema_utils import serialize_schema


HIDDEN_SIZE = 768
ROBERTA_MODEL = "roberta-base"
MAX_SEQ_LEN = 512   # RoBERTa's hard limit — log examples that exceed this


class SchemaAwareEncoder(nn.Module):
    """
    Wraps RoBERTa-base and adds a type embedding layer.

    The forward() method takes:
        input_ids      — tokenized [CLS] question [SEP] schema [SEP]
        attention_mask — 1 for real tokens, 0 for padding
        token_type_ids — 0 for question tokens, 1 for schema tokens

    And returns:
        Q_enc    — [batch, 768]                   the [CLS] hidden state
        S_schema — [batch, n_schema_tokens, 768]  schema token hidden states
    """

    def __init__(self, schema_token_start_indices: list = None):
        """
        Args:
            schema_token_start_indices: not needed at init time — passed during forward()
                                        We keep track of where schema tokens begin per example
                                        by parsing the token_type_ids.
        """
        super().__init__()

        # ── 1. Load RoBERTa backbone ──────────────────────────────
        # CONCEPT: from_pretrained downloads (or loads from cache) the RoBERTa-base
        # weights that were pre-trained on a massive English corpus. We're not
        # training from scratch — we're adapting these existing weights.
        print(f"Loading {ROBERTA_MODEL} backbone...")
        self.roberta = AutoModel.from_pretrained(ROBERTA_MODEL)

        # ── 2. Type embedding layer ───────────────────────────────
        # CONCEPT: nn.Embedding(2, 768) creates a lookup table with 2 rows,
        # each of size 768. Row 0 = "question type vector", Row 1 = "schema type vector".
        # These start as random values and are learned during training.
        # We ADD this vector to each token's base embedding before feeding into RoBERTa.
        self.type_embedding = nn.Embedding(2, HIDDEN_SIZE)

        # Initialize type embeddings to near-zero so they don't disrupt
        # RoBERTa's pretrained representations at the start of training.
        nn.init.normal_(self.type_embedding.weight, mean=0.0, std=0.02)

        print("SchemaAwareEncoder initialized.")
        print(f"  Backbone: {ROBERTA_MODEL}  (hidden size: {HIDDEN_SIZE})")
        print(f"  Type embedding: nn.Embedding(2, {HIDDEN_SIZE})")

    def forward(
        self,
        input_ids: torch.Tensor,      # [batch, seq_len]
        attention_mask: torch.Tensor, # [batch, seq_len]  1=real token, 0=padding
        token_type_ids: torch.Tensor, # [batch, seq_len]  0=question, 1=schema
    ):
        """
        Forward pass through the schema-aware encoder.

        WHAT HAPPENS INSIDE:
            1. RoBERTa looks up its internal token embeddings for input_ids
            2. We add our type embeddings (0 or 1 per token)
            3. The combined embeddings pass through RoBERTa's transformer layers
            4. We extract the [CLS] hidden state as Q_enc
            5. We extract schema token hidden states as S_schema

        Returns:
            Q_enc    : torch.Tensor [batch, 768]
            S_schema : torch.Tensor [batch, n_schema_tokens, 768]
        """
        batch_size, seq_len = input_ids.shape

        # ── Step 1: Get RoBERTa's input embeddings ────────────────
        # CONCEPT: RoBERTa has an internal embedding lookup table (vocab_size × 768).
        # We need to access it BEFORE the transformer layers so we can ADD our
        # type embeddings to the raw token embeddings.
        #
        # inputs_embeds allows us to bypass the internal embedding lookup and
        # directly provide the embeddings — with our type signal already added.
        token_embeds = self.roberta.embeddings.word_embeddings(input_ids)
        # token_embeds shape: [batch, seq_len, 768]

        # ── Step 2: Compute and add type embeddings ───────────────
        # token_type_ids is 0 for question tokens, 1 for schema tokens.
        # self.type_embedding(token_type_ids) looks up the type vector for each token.
        type_embeds = self.type_embedding(token_type_ids)
        # type_embeds shape: [batch, seq_len, 768]

        # Add type embeddings to token embeddings
        # CONCEPT: Addition is how BERT/RoBERTa combine different embedding signals.
        # The model learns to use the type signal during backpropagation.
        combined_embeds = token_embeds + type_embeds
        # combined_embeds shape: [batch, seq_len, 768]

        # ── Step 3: Forward pass through RoBERTa ─────────────────
        # We pass inputs_embeds instead of input_ids so RoBERTa uses our
        # combined embeddings (token + type) instead of doing a fresh lookup.
        outputs = self.roberta(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
        )
        # outputs.last_hidden_state shape: [batch, seq_len, 768]
        all_hidden = outputs.last_hidden_state

        # ── Step 4: Extract Q_enc from [CLS] token ───────────────
        # CONCEPT: RoBERTa's [CLS] token (always at position 0) aggregates
        # information from the entire input sequence via self-attention.
        # By the last transformer layer, it represents the "global meaning"
        # of the entire input — both question and schema together.
        # This is your query representation that Noor uses as part of MDP state.
        Q_enc = all_hidden[:, 0, :]  # [batch, 768]

        # ── Step 5: Extract S_schema from schema token positions ──
        # CONCEPT: We need the hidden states ONLY at the positions where
        # schema tokens are. token_type_ids == 1 marks exactly those positions.
        # We use a boolean mask to gather them.
        #
        # Note:
        #   The number of schema tokens can vary between examples in a batch
        #   (different databases have different numbers of tables/columns).
        #   We use padding to make them the same length within a batch.
        schema_mask = (token_type_ids == 1)  # [batch, seq_len]  boolean

        # Collect schema hidden states per example
        # We find the max number of schema tokens in this batch for padding
        n_schema_per_example = schema_mask.sum(dim=1)  # [batch]
        max_schema_tokens = int(n_schema_per_example.max().item())

        if max_schema_tokens == 0:
            # Edge case: no schema tokens found (shouldn't happen, but guard against it)
            S_schema = torch.zeros(batch_size, 1, HIDDEN_SIZE, device=input_ids.device)
            return Q_enc, S_schema

        # Build S_schema by gathering schema token hidden states
        S_schema = torch.zeros(
            batch_size, max_schema_tokens, HIDDEN_SIZE, device=input_ids.device
        )
        for b in range(batch_size):
            schema_positions = schema_mask[b].nonzero(as_tuple=True)[0]
            n = len(schema_positions)
            if n > 0:
                S_schema[b, :n, :] = all_hidden[b, schema_positions, :]
            # remaining positions in S_schema stay as zero (padding)

        return Q_enc, S_schema

    def get_output_shapes(self, batch_size: int, n_schema_tokens: int) -> dict:
        """
        Utility: returns expected output shapes. Use this to verify with Noor.
        """
        return {
            "Q_enc":    (batch_size, HIDDEN_SIZE),
            "S_schema": (batch_size, n_schema_tokens, HIDDEN_SIZE),
        }


# ─────────────────────────────────────────────────────────────────
# TOKENIZATION HELPER
# ─────────────────────────────────────────────────────────────────

def tokenize_question_and_schema(
    question: str,
    schema_str: str,
    tokenizer,
    max_length: int = MAX_SEQ_LEN,
) -> dict:
    """
    Tokenize a question + schema string together and build token_type_ids.

    CONCEPT:
        HuggingFace's tokenizer handles [CLS] and [SEP] automatically.
        We then walk through the output and label:
            - [CLS] token           → type 0  (treated as question)
            - question tokens       → type 0
            - [SEP] between Q & S   → type 0  (boundary, neutral)
            - schema tokens         → type 1
            - trailing [SEP]        → type 1

    Args:
        question:   raw natural language question string
        schema_str: output of serialize_schema(), e.g. "employees: id(INT), name(TEXT)"
        tokenizer:  HuggingFace AutoTokenizer
        max_length: hard limit (RoBERTa max is 512 tokens)

    Returns:
        dict with keys: input_ids, attention_mask, token_type_ids
        All values are torch.Tensor of shape [seq_len]
    """
    # Tokenize the question alone (no special tokens) to know its length
    q_tokens = tokenizer(
        question,
        add_special_tokens=False,
        return_tensors="pt",
    )
    q_len = q_tokens["input_ids"].shape[1]  # number of question tokens

    # Tokenize question + schema together (tokenizer adds [CLS]...[SEP]...[SEP])
    encoding = tokenizer(
        question,
        schema_str,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].squeeze(0)      # [seq_len]
    attention_mask = encoding["attention_mask"].squeeze(0)  # [seq_len]
    seq_len = input_ids.shape[0]

    # Build token_type_ids manually:
    # Layout:  [CLS] q_tokens [SEP] schema_tokens [SEP] [PAD]...
    # Type:      0      0       0        1           1     0
    token_type_ids = torch.zeros(seq_len, dtype=torch.long)

    # schema starts after: [CLS] (1) + q_tokens (q_len) + [SEP] (1) = q_len + 2
    schema_start = q_len + 2

    # Mark schema tokens as type 1 (up to the trailing [SEP] or [PAD])
    for i in range(schema_start, seq_len):
        if attention_mask[i] == 0:
            break  # hit padding — stop
        token_type_ids[i] = 1

    # Log if this example was truncated (important to monitor!)
    total_tokens_needed = 1 + q_len + 1 + len(
        tokenizer(schema_str, add_special_tokens=False)["input_ids"]
    ) + 1
    if total_tokens_needed > max_length:
        print(f"[TRUNCATION WARNING] Example needs {total_tokens_needed} tokens "
              f"but max is {max_length}. Schema was truncated.")

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }


# ─────────────────────────────────────────────────────────────────
# QUICK VERIFICATION TEST
# Run: python encoder.py
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Task 1 — Encoder Verification Test")
    print("=" * 60)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL)

    # Build encoder
    encoder = SchemaAwareEncoder()
    encoder.eval()

    # Fake example (no Spider download needed for this test)
    question   = "Find all employees with salary above 5000"
    schema_str = "employees: id(INT), name(TEXT), salary(REAL), dept_id(INT) | departments: id(INT), name(TEXT)"

    print(f"\nQuestion:   '{question}'")
    print(f"Schema:     '{schema_str}'")

    # Tokenize
    tokens = tokenize_question_and_schema(question, schema_str, tokenizer)

    print(f"\nToken shapes:")
    print(f"  input_ids:      {tokens['input_ids'].shape}")
    print(f"  attention_mask: {tokens['attention_mask'].shape}")
    print(f"  token_type_ids: {tokens['token_type_ids'].shape}")

    # Show token breakdown
    ids_list   = tokens["input_ids"].tolist()
    type_list  = tokens["token_type_ids"].tolist()
    word_list  = tokenizer.convert_ids_to_tokens(ids_list)

    print("\nFirst 20 tokens (word | type):")
    for i, (w, t) in enumerate(zip(word_list[:20], type_list[:20])):
        label = "QUESTION" if t == 0 else "SCHEMA"
        print(f"  [{i:2d}]  {w:20s}  type={t}  ({label})")

    q_count = type_list.count(0)
    s_count = type_list.count(1)
    print(f"\nQuestion tokens (type=0): {q_count}")
    print(f"Schema tokens   (type=1): {s_count}")

    # Forward pass
    input_ids      = tokens["input_ids"].unsqueeze(0)       # add batch dim
    attention_mask = tokens["attention_mask"].unsqueeze(0)
    token_type_ids = tokens["token_type_ids"].unsqueeze(0)

    print("\nRunning forward pass...")
    with torch.no_grad():
        Q_enc, S_schema = encoder(input_ids, attention_mask, token_type_ids)

    print(f"\n✓ Q_enc shape:    {Q_enc.shape}    (expected: [1, 768])")
    print(f"✓ S_schema shape: {S_schema.shape}  (expected: [1, {s_count}, 768])")

    assert Q_enc.shape == (1, 768), f"Q_enc shape mismatch! Got {Q_enc.shape}"
    assert S_schema.shape[0] == 1 and S_schema.shape[2] == 768

    print("\n✓ All shapes correct. Task 1 encoder is working.")
    print("\n--- Tell Noor ---")
    print(f"Q_enc shape:    {list(Q_enc.shape)}   <- use for MDP state")
    print(f"S_schema shape: {list(S_schema.shape)} <- use for schema cross-attention")
    print("Schema tokens are in order: all cols of table1, then table2, etc.")