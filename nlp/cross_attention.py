"""
cross_attention.py
------------------
Cross-Attention Schema Linking module for Text-to-SQL.

WHAT THIS FILE DOES:
    Takes H_q (question hidden states) and H_s (schema hidden states)
    from the encoder and produces:
        1. attended_schema  — each question token enriched with schema info
        2. alignment_map    — structured dict exported for heatmap / demo
        3. flat_scores      — compact [n_schema] vector for Noor's MDP state

THE MATH (one forward pass):
    Q_proj = Linear(768, 64)(H_q)          [batch, n_q,     64]
    K_proj = Linear(768, 64)(H_s)          [batch, n_schema, 64]
    scores = Q_proj @ K_proj.T / 8.0       [batch, n_q,     n_schema]
    A      = softmax(scores, dim=-1)        [batch, n_q,     n_schema]
    attended_schema = A @ H_s              [batch, n_q,     768]
    flat_scores     = A.max(dim=1).values  [batch, n_schema]

INTEGRATION WITH NOOR:
    flat_scores  → plug directly into MDP state encoder
    alignment_map → rendered as heatmap in frontend (Task 9)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict


# ── projection dimension ──────────────────────────────────────────
# 64 = 768 / 12  (matches RoBERTa's per-head dimension)
# Reduces compute 12x vs operating in full 768-dim space.
PROJ_DIM   = 64
HIDDEN_DIM = 768


class CrossAttentionSchemaLinker(nn.Module):
    """
    Dedicated cross-attention module for schema grounding.

    Takes question and schema hidden states from the encoder,
    produces alignment scores and enriched question representations.

    Why a separate module and not just RoBERTa's internal attention?
        RoBERTa's internal heads do many jobs simultaneously.
        This module has ONE job: map question tokens to schema elements.
        It also exports explicit alignment scores in the format Noor needs.
    """

    def __init__(self):
        super().__init__()

        # ── learnable projection layers ───────────────────────────
        # CONCEPT: instead of comparing 768-dim vectors directly
        # (expensive, noisy), we project both Q and K down to 64-dim.
        # These projections are LEARNED — the model figures out which
        # 64 dimensions are most useful for schema matching.
        #
        # Note: V does NOT get projected — we keep it at 768-dim
        # so the output attended_schema preserves full information.
        self.W_q = nn.Linear(HIDDEN_DIM, PROJ_DIM, bias=False)
        self.W_k = nn.Linear(HIDDEN_DIM, PROJ_DIM, bias=False)

        # scaling factor: 1 / sqrt(PROJ_DIM) = 1/8
        # prevents dot products from growing too large → softmax stays smooth
        self.scale = PROJ_DIM ** -0.5   # 0.125

        # ── attention dropout ─────────────────────────────────────
        # applied to attention weights during training to prevent
        # the module from over-relying on a single schema element
        self.attn_dropout = nn.Dropout(p=0.1)

        print("CrossAttentionSchemaLinker initialized.")
        print(f"  Q projection: Linear({HIDDEN_DIM}, {PROJ_DIM})")
        print(f"  K projection: Linear({HIDDEN_DIM}, {PROJ_DIM})")
        print(f"  Scale factor: {self.scale:.4f}  (= 1/sqrt({PROJ_DIM}))")

    def forward(
        self,
        H_q:           torch.Tensor,        # [batch, n_q,     768]
        H_s:           torch.Tensor,        # [batch, n_schema, 768]
        q_mask:        torch.Tensor = None, # [batch, n_q]     1=real, 0=pad
        schema_mask:   torch.Tensor = None, # [batch, n_schema] 1=real, 0=pad
    ):
        """
        Forward pass — compute cross-attention between question and schema.

        Args:
            H_q:         question token hidden states from encoder
            H_s:         schema element hidden states from encoder
            q_mask:      optional mask for padded question tokens
            schema_mask: optional mask for padded schema tokens

        Returns:
            attended_schema  [batch, n_q, 768]      enriched question states
            A                [batch, n_q, n_schema]  attention weights (alignment)
            flat_scores      [batch, n_schema]        max score per schema element
        """
        # ── Step 1: project Q and K to 64-dim ─────────────────────
        # CONCEPT: W_q and W_k are learned linear transformations.
        # They compress 768 → 64 while preserving the dimensions
        # most useful for computing question-schema similarity.
        Q_proj = self.W_q(H_q)   # [batch, n_q,     64]
        K_proj = self.W_k(H_s)   # [batch, n_schema, 64]

        # ── Step 2: compute raw attention scores ──────────────────
        # Q_proj @ K_proj.transpose(-1,-2):
        #   for every (question token i, schema element j) pair,
        #   compute dot product of their 64-dim projections.
        #
        # Shape: [batch, n_q, 64] @ [batch, 64, n_schema]
        #      = [batch, n_q, n_schema]
        scores = torch.bmm(Q_proj, K_proj.transpose(1, 2)) * self.scale
        # scores[b, i, j] = how relevant is schema element j to question token i

        # ── Step 3: mask padding positions before softmax ─────────
        # CONCEPT: padded schema tokens should never receive attention.
        # We set their scores to -inf so softmax gives them prob ≈ 0.
        if schema_mask is not None:
            # schema_mask: [batch, n_schema] → expand to [batch, 1, n_schema]
            mask = schema_mask.unsqueeze(1).bool()
            scores = scores.masked_fill(~mask, float("-inf"))

        # ── Step 4: softmax over schema dimension ─────────────────
        # dim=-1 means we normalise across schema elements for each
        # question token. Each row (question token) sums to 1.0.
        A = F.softmax(scores, dim=-1)   # [batch, n_q, n_schema]

        # Replace NaN from all-masked rows (full padding edge case)
        A = torch.nan_to_num(A, nan=0.0)

        # Apply dropout during training
        A = self.attn_dropout(A)

        # ── Step 5: attended_schema = A @ H_s ────────────────────
        # CONCEPT: weighted sum of schema vectors.
        # A[b, i, :] tells us how much question token i attends to
        # each schema element. We blend H_s vectors by those weights.
        #
        # Shape: [batch, n_q, n_schema] @ [batch, n_schema, 768]
        #      = [batch, n_q, 768]
        #
        # attended_schema[b, i, :] = sum_j( A[b,i,j] * H_s[b,j,:] )
        # Each question token is now enriched with schema information.
        attended_schema = torch.bmm(A, H_s)   # [batch, n_q, 768]

        # ── Step 6: flat_scores — max per schema element ──────────
        # For each schema element, take the MAXIMUM attention score
        # it received from ANY question token.
        # This gives one number per schema element:
        # "how relevant is this column/table to the question overall?"
        # Shape: [batch, n_q, n_schema] → max over n_q → [batch, n_schema]
        flat_scores = A.max(dim=1).values   # [batch, n_schema]

        return attended_schema, A, flat_scores


# ─────────────────────────────────────────────────────────────────
# ALIGNMENT MAP BUILDER
#
# Converts the raw attention tensor A into the structured dict
# format that the frontend heatmap and Noor's state encoder use.
# ─────────────────────────────────────────────────────────────────

def build_alignment_map(
    A:              torch.Tensor,   # [n_q, n_schema]  (single example, no batch dim)
    question_tokens: List[str],     # decoded question token strings
    schema_names:    List[str],     # schema element names e.g. ["employees.id", "employees.salary"]
    threshold:       float = 0.1,   # only include links with score above this
) -> List[Dict]:
    """
    Convert attention matrix to structured alignment map.

    CONCEPT:
        The alignment map is what you export for the demo heatmap and
        for Noor's MDP. For each question token, it lists which schema
        elements it attends to (above threshold) and by how much.

    Args:
        A:               attention weights [n_q, n_schema] for ONE example
        question_tokens: list of decoded token strings for the question
        schema_names:    list of schema element names in order
        threshold:       minimum score to include in output

    Returns:
        List of dicts, one per question token that has at least one
        schema link above the threshold.

    Format (matches the contract with Noor):
        [
            {
                "token_idx":  int,
                "token_text": str,
                "links": [
                    {"schema_id": int, "schema_name": str, "score": float},
                    ...  sorted by score descending
                ]
            },
            ...
        ]
    """
    alignment_map = []
    A_np = A.detach().cpu().numpy()

    for tok_i, tok_text in enumerate(question_tokens):
        if tok_i >= A_np.shape[0]:
            break

        # get this token's attention scores over all schema elements
        scores = A_np[tok_i]   # [n_schema]

        # filter to only links above threshold
        links = []
        for schema_j, score in enumerate(scores):
            if schema_j >= len(schema_names):
                break
            if float(score) >= threshold:
                links.append({
                    "schema_id":   schema_j,
                    "schema_name": schema_names[schema_j],
                    "score":       round(float(score), 4),
                })

        # only include this token if it has at least one strong link
        if links:
            # sort by score descending — strongest link first
            links.sort(key=lambda x: x["score"], reverse=True)
            alignment_map.append({
                "token_idx":  tok_i,
                "token_text": tok_text,
                "links":      links,
            })

    return alignment_map


def build_schema_names(db_id: str, schema_dict: dict) -> List[str]:
    """
    Build a list of schema element names in the same order as S_schema.

    CONCEPT:
        S_schema rows are ordered as: all columns of table1, then table2, etc.
        This function builds the matching name list so you can label each row.

    Returns:
        ["employees.Perpetrator_ID", "employees.People_ID", ..., "people.Name", ...]
    """
    entry  = schema_dict[db_id]
    tables = entry["tables"]
    cols   = entry["columns"]   # [[col_name, col_type, table_idx], ...]

    # group columns by table in order
    table_cols = {t: [] for t in tables}
    for col_name, col_type, t_idx in cols:
        table_name = tables[t_idx]
        table_cols[table_name].append(f"{table_name}.{col_name}")

    # flatten in table order — matches S_schema row order
    names = []
    for t in tables:
        names.extend(table_cols[t])
    return names


# ─────────────────────────────────────────────────────────────────
# VERIFICATION TEST
# Run: python cross_attention.py
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from transformers import AutoTokenizer
    from config import TABLES_JSON
    from schema_utils import load_schema_dict, serialize_schema
    from encoder import SchemaAwareEncoder

    print("=" * 60)
    print("Task 3 — Cross-Attention Schema Linker Verification")
    print("=" * 60)

    # ── setup ─────────────────────────────────────────────────────
    tokenizer   = AutoTokenizer.from_pretrained("roberta-base")
    schema_dict = load_schema_dict(TABLES_JSON)
    encoder     = SchemaAwareEncoder()
    linker      = CrossAttentionSchemaLinker()
    encoder.eval()
    linker.eval()

    # ── pick a good demo example ──────────────────────────────────
    db_id    = "perpetrator"
    question = "How many perpetrators were killed?"

    print(f"\nQuestion: '{question}'")
    print(f"Database: {db_id}")

    # ── prepare input ─────────────────────────────────────────────
    schema_str = serialize_schema(db_id, schema_dict)
    print(f"Schema:   {schema_str[:80]}...")

    q_tokens_raw = tokenizer(question, add_special_tokens=False)["input_ids"]
    q_len        = len(q_tokens_raw)

    encoding = tokenizer(
        question, schema_str,
        add_special_tokens=True,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids      = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    # build token_type_ids
    schema_start   = q_len + 2
    seq_len        = input_ids.shape[1]
    token_type_ids = torch.zeros(1, seq_len, dtype=torch.long)
    for i in range(schema_start, seq_len):
        if attention_mask[0, i] == 0:
            break
        token_type_ids[0, i] = 1

    # ── encoder forward pass ──────────────────────────────────────
    print("\nRunning encoder...")
    with torch.no_grad():
        Q_enc, S_schema, all_hidden = encoder(input_ids, attention_mask, token_type_ids)

    # extract H_q — question token hidden states (positions 1 to q_len)
    # position 0 = [CLS], positions 1..q_len = question tokens
    
    _, _, all_hidden = encoder(input_ids, attention_mask, token_type_ids)
    H_q              = all_hidden[:, 1:q_len+1, :]

    print(f"H_q shape:    {H_q.shape}      (question hidden states)")
    print(f"S_schema shape: {S_schema.shape}  (schema hidden states)")

    # ── cross-attention forward pass ──────────────────────────────
    print("\nRunning cross-attention linker...")
    with torch.no_grad():
        attended_schema, A, flat_scores = linker(H_q, S_schema)

    print(f"\n✓ attended_schema: {attended_schema.shape}   (expected [1, n_q, 768])")
    print(f"✓ A (alignment):   {A.shape}   (expected [1, n_q, n_schema])")
    print(f"✓ flat_scores:     {flat_scores.shape}     (expected [1, n_schema])")

    # ── build alignment map ───────────────────────────────────────
    q_token_strings = tokenizer.convert_ids_to_tokens(
        input_ids[0, 1:q_len+1].tolist()
    )
    q_token_strings = [t.lstrip("Ġ") for t in q_token_strings]

    schema_names = build_schema_names(db_id, schema_dict)

    alignment_map = build_alignment_map(
        A[0],            # remove batch dim
        q_token_strings,
        schema_names,
        threshold=0.1,
    )

    # ── print alignment map ───────────────────────────────────────
    print(f"\n--- Alignment Map (threshold=0.1) ---")
    print(f"{'Question token':<18} → Top schema links")
    print("-" * 60)
    for entry in alignment_map:
        top = entry["links"][:3]   # top 3 links per token
        links_str = "  |  ".join(
            f"{l['schema_name']} ({l['score']:.3f})" for l in top
        )
        print(f"  {entry['token_text']:<16} → {links_str}")

    # ── print flat scores for Noor ────────────────────────────────
    print(f"\n--- Flat scores (tell Noor) ---")
    print(f"Shape: {flat_scores.shape}  — one score per schema element")
    fs = flat_scores[0].tolist()
    for name, score in sorted(zip(schema_names, fs), key=lambda x: -x[1])[:5]:
        print(f"  {name:<35} {score:.4f}")

    print("\n✓ Task 3 cross-attention schema linker working correctly.")
    print("  Alignment map ready for heatmap visualization.")
    print("  flat_scores ready for Noor's MDP state encoder.")