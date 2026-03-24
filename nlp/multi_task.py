"""
multi_task.py
-------------
Multi-Task Learning model combining SQL generation and schema
element classification into one joint training objective.

ARCHITECTURE:
    SchemaAwareEncoder  (Task 1) — shared backbone
         |
         ├── SQL decoder head    — generates SQL token by token
         |       loss: L_sql   = cross_entropy(logits, gold_sql[1:])
         |
         └── Schema classif. head — classifies each input token
                 loss: L_schema = cross_entropy(cls_logits, token_labels,
                                                ignore_index=-100)

    L_total = L_sql + lambda * L_schema     (lambda = 0.1)

CRITICAL NOTE FOR NOOR:
    The schema_cls_head is ONLY used during supervised pretraining.
    When loading this checkpoint for RL fine-tuning, load ONLY:
        encoder_state_dict
        decoder_state_dict
    DO NOT load schema_cls_head_state_dict — it is saved separately
    under its own key so it can be easily ignored.

WHY MULTI-TASK LEARNING WORKS:
    The schema classification loss forces the encoder to produce
    hidden states where TABLE / COLUMN / VALUE tokens are cleanly
    separable. This structural signal makes the representations
    richer for SQL generation too — the encoder learns what kind
    of token it is processing, not just what the token means.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nlp.encoder import SchemaAwareEncoder, HIDDEN_SIZE

# ── label constants (must match data_pipeline.py) ────────────────
TABLE  = 0
COLUMN = 1
VALUE  = 2
NONE   = 3
IGNORE = -100   # cross_entropy ignore_index


class TextToSQLModel(nn.Module):
    """
    Full Text-to-SQL model with dual training objective.

    Components:
        encoder          — SchemaAwareEncoder (Task 1)
        schema_cls_head  — Linear(768, 4) for token classification
        decoder          — simple Transformer decoder for SQL generation

    During training:
        forward() returns L_total, L_sql, L_schema
        All three are logged separately every 100 steps.

    During inference:
        call encode() to get encoder hidden states
        call decode_step() to generate one SQL token at a time
    """

    def __init__(
        self,
        vocab_size:   int   = 50265,   # RoBERTa vocab size
        decoder_layers: int = 3,
        decoder_heads:  int = 8,
        decoder_ff_dim: int = 1024,
        max_sql_len:    int = 128,
        mtl_lambda:     float = 0.1,
    ):
        super().__init__()

        self.vocab_size  = vocab_size
        self.mtl_lambda  = mtl_lambda
        self.max_sql_len = max_sql_len

        # ── encoder (Task 1) ──────────────────────────────────────
        self.encoder = SchemaAwareEncoder()

        # ── schema classification head ────────────────────────────
        # CONCEPT: takes encoder hidden states [batch, seq_len, 768]
        # and projects each token position to 4 class scores.
        # Applied to ALL input tokens during training.
        # NOT used during inference / RL fine-tuning.
        self.schema_cls_head = nn.Linear(HIDDEN_SIZE, 4)

        # ── SQL decoder ───────────────────────────────────────────
        # CONCEPT: a Transformer decoder that generates SQL tokens
        # one at a time. It attends to its own partial output (self-
        # attention) and to the encoder hidden states (cross-attention).
        #
        # We use PyTorch's built-in TransformerDecoder for simplicity.
        # Input to decoder: gold_sql[:-1] token embeddings
        # Output:           logits over vocab at each position
        self.token_embedding = nn.Embedding(vocab_size, HIDDEN_SIZE)
        self.pos_embedding   = nn.Embedding(max_sql_len, HIDDEN_SIZE)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=HIDDEN_SIZE,
            nhead=decoder_heads,
            dim_feedforward=decoder_ff_dim,
            dropout=0.1,
            batch_first=True,   # [batch, seq, dim] instead of [seq, batch, dim]
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=decoder_layers,
        )

        # final projection: decoder hidden states → vocab logits
        self.output_projection = nn.Linear(HIDDEN_SIZE, vocab_size)

        print("TextToSQLModel initialized.")
        print(f"  Encoder:          SchemaAwareEncoder (RoBERTa-base)")
        print(f"  Decoder layers:   {decoder_layers}")
        print(f"  Decoder heads:    {decoder_heads}")
        print(f"  Schema cls head:  Linear({HIDDEN_SIZE}, 4)")
        print(f"  MTL lambda:       {mtl_lambda}")

    def forward(
        self,
        input_ids:      torch.Tensor,   # [batch, seq_len]
        attention_mask: torch.Tensor,   # [batch, seq_len]
        token_type_ids: torch.Tensor,   # [batch, seq_len]
        gold_sql_ids:   torch.Tensor,   # [batch, sql_len]
        token_labels:   torch.Tensor,   # [batch, seq_len]
    ):
        """
        Full forward pass — encoder + both heads + combined loss.

        TEACHER FORCING:
            Decoder receives gold_sql[:-1] as input and is trained
            to predict gold_sql[1:] at each step.
            e.g. input:  [BOS, SELECT, count, *]
                 target: [SELECT, count, *, FROM]
            The model always sees the correct previous token —
            not its own prediction. This stabilises training.

        Returns:
            loss        scalar  L_total = L_sql + lambda * L_schema
            L_sql       scalar  SQL generation loss (main)
            L_schema    scalar  schema classification loss (auxiliary)
            sql_logits  [batch, sql_len-1, vocab_size]  for logging
            cls_logits  [batch, seq_len, 4]             for logging
        """
        batch_size = input_ids.shape[0]

        # ── Step 1: encode question + schema ─────────────────────
        # Q_enc and S_schema come from Task 1 encoder.
        # all_hidden is what we need for both heads.
        Q_enc, S_schema = self.encoder(
            input_ids, attention_mask, token_type_ids
        )

        # get full hidden states for schema classification head
        # CONCEPT: we need hidden states at ALL positions, not just
        # [CLS] and schema positions. Re-run the embedding + RoBERTa
        # forward to get all_hidden.
        token_embeds   = self.encoder.roberta.embeddings.word_embeddings(input_ids)
        type_embeds    = self.encoder.type_embedding(token_type_ids)
        combined       = token_embeds + type_embeds
        encoder_output = self.encoder.roberta(
            inputs_embeds=combined,
            attention_mask=attention_mask,
        )
        all_hidden = encoder_output.last_hidden_state  # [batch, seq_len, 768]

        # ── Step 2: schema classification head ───────────────────
        # Apply the classification head to every token position.
        # cls_logits[b, i, :] = 4 class scores for token i in example b
        cls_logits = self.schema_cls_head(all_hidden)  # [batch, seq_len, 4]

        # ── Step 3: SQL decoder (teacher forcing) ────────────────
        # CONCEPT:
        #   gold_sql_ids shape: [batch, sql_len]
        #   Decoder input:  gold_sql_ids[:, :-1]  (all but last token)
        #   Decoder target: gold_sql_ids[:, 1:]   (all but first token)
        #
        # This way at position t, the decoder sees tokens 0..t-1
        # and must predict token t.
        decoder_input  = gold_sql_ids[:, :-1]   # [batch, sql_len-1]
        decoder_target = gold_sql_ids[:, 1:]    # [batch, sql_len-1]

        sql_len = decoder_input.shape[1]

        # embed decoder input tokens + positions
        positions  = torch.arange(sql_len, device=input_ids.device).unsqueeze(0)
        dec_embeds = (
            self.token_embedding(decoder_input)
            + self.pos_embedding(positions)
        )   # [batch, sql_len-1, 768]

        # causal mask: decoder can only attend to previous positions
        # (can't look at future tokens — this is autoregressive)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            sql_len, device=input_ids.device
        )

        # encoder padding mask for decoder cross-attention
        # True = position is padding (should be ignored)
        enc_pad_mask = (attention_mask == 0)   # [batch, seq_len]

        # decoder forward pass
        # CONCEPT: decoder cross-attends to all_hidden (encoder output)
        # while autoregressively generating SQL tokens.
        dec_hidden = self.decoder(
            tgt=dec_embeds,                   # [batch, sql_len-1, 768]
            memory=all_hidden,                # [batch, seq_len,   768]
            tgt_mask=causal_mask,             # causal mask
            memory_key_padding_mask=enc_pad_mask,  # ignore encoder padding
        )   # [batch, sql_len-1, 768]

        # project to vocabulary
        sql_logits = self.output_projection(dec_hidden)  # [batch, sql_len-1, vocab_size]

        # ── Step 4: compute losses ────────────────────────────────

        # L_sql: cross-entropy between predicted logits and gold SQL
        # reshape: [batch * (sql_len-1), vocab_size] vs [batch * (sql_len-1)]
        L_sql = F.cross_entropy(
            sql_logits.reshape(-1, self.vocab_size),
            decoder_target.reshape(-1),
            ignore_index=1,   # RoBERTa's <pad> token id = 1
        )

        # L_schema: cross-entropy between cls_logits and token_labels
        # ignore_index=-100 skips [CLS], [SEP], [PAD] tokens
        # reshape: [batch * seq_len, 4] vs [batch * seq_len]
        L_schema = F.cross_entropy(
            cls_logits.reshape(-1, 4),
            token_labels.reshape(-1),
            ignore_index=IGNORE,
        )

        # combined loss
        L_total = L_sql + self.mtl_lambda * L_schema

        return L_total, L_sql, L_schema, sql_logits, cls_logits

    def encode(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ):
        """
        Encode only — used during inference and RL fine-tuning.
        Returns all_hidden for use by the decoder step by step.
        """
        token_embeds   = self.encoder.roberta.embeddings.word_embeddings(input_ids)
        type_embeds    = self.encoder.type_embedding(token_type_ids)
        combined       = token_embeds + type_embeds
        encoder_output = self.encoder.roberta(
            inputs_embeds=combined,
            attention_mask=attention_mask,
        )
        return encoder_output.last_hidden_state   # [batch, seq_len, 768]

    def save_checkpoint(self, path: str, extra: dict = None):
        """
        Save checkpoint with clearly separated keys.

        TELL NOOR: load encoder_state_dict + decoder_state_dict only.
        Do NOT load schema_cls_head_state_dict during RL fine-tuning.
        """
        ckpt = {
            "encoder_state_dict":          self.encoder.state_dict(),
            "decoder_state_dict":          self.decoder.state_dict(),
            "output_projection_state_dict": self.output_projection.state_dict(),
            "token_embedding_state_dict":  self.token_embedding.state_dict(),
            "pos_embedding_state_dict":    self.pos_embedding.state_dict(),
            "schema_cls_head_state_dict":  self.schema_cls_head.state_dict(),  # Noor ignores this
        }
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, path)
        print(f"Checkpoint saved → {path}")

    @classmethod
    def load_for_rl(cls, path: str, **kwargs):
        """
        Load checkpoint for RL fine-tuning (Noor uses this).
        Loads encoder + decoder only — skips schema_cls_head.
        """
        model = cls(**kwargs)
        ckpt  = torch.load(path, map_location="cpu")
        model.encoder.load_state_dict(ckpt["encoder_state_dict"])
        model.decoder.load_state_dict(ckpt["decoder_state_dict"])
        model.output_projection.load_state_dict(ckpt["output_projection_state_dict"])
        model.token_embedding.load_state_dict(ckpt["token_embedding_state_dict"])
        model.pos_embedding.load_state_dict(ckpt["pos_embedding_state_dict"])
        # schema_cls_head intentionally NOT loaded
        print(f"RL checkpoint loaded from {path} (schema_cls_head skipped)")
        return model


# ─────────────────────────────────────────────────────────────────
# VERIFICATION TEST
# Run: python multi_task.py
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import TRAIN_JSON, DEV_JSON, TABLES_JSON
    from data_pipeline import build_dataloaders

    print("=" * 60)
    print("Task 4 — Multi-Task Model Verification")
    print("=" * 60)

    # ── build one batch ───────────────────────────────────────────
    print("\nLoading one batch from Spider...")
    train_loader, _, _, _ = build_dataloaders(
        TRAIN_JSON, DEV_JSON, TABLES_JSON,
        batch_size=2,    # tiny batch for quick test
    )
    batch = next(iter(train_loader))

    input_ids      = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    token_type_ids = batch["token_type_ids"]
    gold_sql_ids   = batch["gold_sql_ids"]
    token_labels   = batch["token_labels"]

    print(f"Batch shapes:")
    print(f"  input_ids:    {input_ids.shape}")
    print(f"  gold_sql_ids: {gold_sql_ids.shape}")
    print(f"  token_labels: {token_labels.shape}")

    # ── build model ───────────────────────────────────────────────
    print("\nBuilding TextToSQLModel...")
    model = TextToSQLModel()
    model.train()

    # ── forward pass ──────────────────────────────────────────────
    print("\nRunning forward pass...")
    L_total, L_sql, L_schema, sql_logits, cls_logits = model(
        input_ids, attention_mask, token_type_ids,
        gold_sql_ids, token_labels,
    )

    print(f"\n✓ L_total:    {L_total.item():.4f}")
    print(f"✓ L_sql:      {L_sql.item():.4f}")
    print(f"✓ L_schema:   {L_schema.item():.4f}")
    print(f"✓ sql_logits: {sql_logits.shape}   (expected [2, sql_len-1, vocab_size])")
    print(f"✓ cls_logits: {cls_logits.shape}    (expected [2, seq_len, 4])")

    # ── verify combined loss formula ──────────────────────────────
    expected = L_sql.item() + 0.1 * L_schema.item()
    assert abs(L_total.item() - expected) < 1e-4, "Loss formula wrong!"
    print(f"\n✓ L_total = L_sql + 0.1 × L_schema  verified")
    print(f"  {L_total.item():.4f} = {L_sql.item():.4f} + 0.1 × {L_schema.item():.4f}")

    # ── verify backward pass ──────────────────────────────────────
    print("\nRunning backward pass...")
    L_total.backward()
    print("✓ Backward pass completed — gradients flow through both heads")

    # check type embedding got gradients
    grad = model.encoder.type_embedding.weight.grad
    assert grad is not None, "Type embedding has no gradient!"
    print(f"✓ type_embedding gradient norm: {grad.norm().item():.4f}")

    # check schema_cls_head got gradients
    grad_cls = model.schema_cls_head.weight.grad
    assert grad_cls is not None, "schema_cls_head has no gradient!"
    print(f"✓ schema_cls_head gradient norm: {grad_cls.norm().item():.4f}")

    print("\n✓ Task 4 multi-task model working correctly.")
    print("  Both losses computing, backward pass clean.")
    print("  Ready for Task 6 (supervised pretraining loop).")