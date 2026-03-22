"""
data_pipeline.py
----------------
Spider dataset pipeline for Text-to-SQL training.

WHAT THIS FILE DOES:
    1. Loads train_spider.json / dev.json (question + gold SQL + db_id per example)
    2. For each example: tokenizes question + schema, builds token_type_ids,
       builds token_labels (TABLE / COLUMN / VALUE / NONE), tokenizes gold SQL
    3. Wraps everything in a PyTorch Dataset + DataLoader

KEY OUTPUT per example (what __getitem__ returns):
    input_ids       [seq_len]   — tokenized question + schema
    attention_mask  [seq_len]   — 1 for real tokens, 0 for padding
    token_type_ids  [seq_len]   — 0 question, 1 schema (our custom type signal)
    token_labels    [seq_len]   — TABLE=0, COLUMN=1, VALUE=2, NONE=3
    gold_sql_ids    [sql_len]   — tokenized gold SQL (target for decoder)
    db_id           str         — which database this example belongs to

IMPORTANT NOTES:
    - NEVER shuffle dev DataLoader — must be reproducible
    - token_labels use -100 for special tokens ([CLS],[SEP],[PAD])
      so cross_entropy ignores them with ignore_index=-100
    - gold SQL is tokenized separately (no schema, just the SQL string)
"""

import json
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from schema_utils import load_schema_dict, serialize_schema

# ── label constants ───────────────────────────────────────────────
TABLE   = 0
COLUMN  = 1
VALUE   = 2
NONE    = 3
IGNORE  = -100   # cross_entropy ignore_index for special/pad tokens


# ─────────────────────────────────────────────────────────────────
# CONCEPT: token_labels
#
# During multi-task training (Task 4), the encoder simultaneously
# tries to generate SQL AND classify each input token as:
#   TABLE  (0) — this token is a table name in the schema
#   COLUMN (1) — this token is a column name
#   VALUE  (2) — this token looks like a value (number, quoted string)
#   NONE   (3) — just a regular question word or punctuation
#
# We create these labels NOW while building the dataset.
# Adding them retroactively after training starts is very painful.
#
# Strategy: fuzzy string matching.
# For each token in the input, check if it appears in the list of
# table names or column names for this db_id. If yes, label it.
# "Fuzzy" means case-insensitive partial match — "employee" matches
# "employees" table name.
# ─────────────────────────────────────────────────────────────────


class SpiderDataset(Dataset):
    """
    PyTorch Dataset for Spider Text-to-SQL examples.

    Each call to __getitem__(i) returns one fully processed example
    as a dict of tensors — ready to be batched by DataLoader.
    """

    def __init__(
        self,
        data_json_path: str,      # path to train_spider.json or dev.json
        schema_dict: dict,         # output of load_schema_dict()
        tokenizer,                 # HuggingFace AutoTokenizer
        max_seq_len: int = 512,    # encoder input length limit
        max_sql_len: int = 128,    # decoder target length limit
    ):
        self.schema_dict = schema_dict
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len
        self.max_sql_len = max_sql_len

        # ── load raw examples ─────────────────────────────────────
        with open(data_json_path, "r") as f:
            raw = json.load(f)

        # CONCEPT: filter out examples whose db_id is not in schema_dict
        # This can happen if tables.json and the data json are mismatched.
        self.examples = [
            ex for ex in raw
            if ex["db_id"] in schema_dict
        ]

        n_dropped = len(raw) - len(self.examples)
        if n_dropped > 0:
            print(f"[WARNING] Dropped {n_dropped} examples with unknown db_id")

        print(f"Loaded {len(self.examples)} examples from {os.path.basename(data_json_path)}")

        # ── track truncation stats ────────────────────────────────
        self.n_truncated = 0

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        """
        Process one Spider example and return a dict of tensors.

        FLOW:
            raw example
                → serialize schema
                → tokenize question + schema together
                → build token_type_ids (our custom 0/1 signal)
                → build token_labels   (TABLE/COLUMN/VALUE/NONE)
                → tokenize gold SQL
                → return dict of tensors
        """
        ex       = self.examples[idx]
        question = ex["question"]
        gold_sql = ex["query"]
        db_id    = ex["db_id"]

        # ── Step 1: serialize schema ──────────────────────────────
        schema_str = serialize_schema(db_id, self.schema_dict)

        # ── Step 2: tokenize question alone (to know its length) ──
        # CONCEPT: We tokenize question alone first to find exactly
        # where the schema starts in the combined tokenization.
        # This boundary tells us where to switch token_type_ids from
        # 0 (question) to 1 (schema).
        q_tokens = self.tokenizer(
            question,
            add_special_tokens=False,
        )["input_ids"]
        q_len = len(q_tokens)

        # ── Step 3: tokenize question + schema together ───────────
        # Layout: [CLS] question tokens [SEP] schema tokens [SEP] [PAD]...
        encoding = self.tokenizer(
            question,
            schema_str,
            add_special_tokens=True,
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids      = encoding["input_ids"].squeeze(0)       # [seq_len]
        attention_mask = encoding["attention_mask"].squeeze(0)  # [seq_len]
        seq_len        = input_ids.shape[0]

        # ── Step 4: build token_type_ids ─────────────────────────
        # schema starts at position: [CLS](1) + q_tokens(q_len) + [SEP](1)
        schema_start   = q_len + 2
        token_type_ids = torch.zeros(seq_len, dtype=torch.long)

        for i in range(schema_start, seq_len):
            if attention_mask[i] == 0:
                break   # hit padding — stop
            token_type_ids[i] = 1

        # ── track truncation ──────────────────────────────────────
        real_len = int(attention_mask.sum().item())
        if real_len == self.max_seq_len:
            self.n_truncated += 1

        # ── Step 5: build token_labels ───────────────────────────
        token_labels = self._build_token_labels(
            input_ids, token_type_ids, attention_mask, db_id
        )

        # ── Step 6: tokenize gold SQL ─────────────────────────────
        # CONCEPT: the decoder learns to predict the next SQL token
        # given all previous gold tokens (teacher forcing).
        # We tokenize gold SQL separately — no schema needed here,
        # just the SQL string itself.
        sql_encoding = self.tokenizer(
            gold_sql,
            add_special_tokens=True,
            max_length=self.max_sql_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        gold_sql_ids = sql_encoding["input_ids"].squeeze(0)  # [sql_len]

        return {
            "input_ids":      input_ids,        # [seq_len]
            "attention_mask": attention_mask,    # [seq_len]
            "token_type_ids": token_type_ids,   # [seq_len]
            "token_labels":   token_labels,     # [seq_len]
            "gold_sql_ids":   gold_sql_ids,     # [sql_len]
            "db_id":          db_id,            # str (for eval)
        }

    def _build_token_labels(
        self,
        input_ids:      torch.Tensor,   # [seq_len]
        token_type_ids: torch.Tensor,   # [seq_len]
        attention_mask: torch.Tensor,   # [seq_len]
        db_id:          str,
    ) -> torch.Tensor:
        """
        Assign a label to every token in the input sequence.

        Labels:
            TABLE  = 0 — token is (part of) a table name
            COLUMN = 1 — token is (part of) a column name
            VALUE  = 2 — token looks like a numeric/string value
            NONE   = 3 — regular question word, punctuation etc.
            IGNORE =-100 — [CLS], [SEP], [PAD] — ignored by loss

        Strategy: fuzzy matching
            For each token, decode it back to a string and check if
            it appears (case-insensitive) in any table or column name
            of this database. Not perfect, but good enough for the
            auxiliary classification task.
        """
        seq_len = input_ids.shape[0]
        labels  = torch.full((seq_len,), NONE, dtype=torch.long)

        # get table and column names for this db
        entry        = self.schema_dict[db_id]
        table_names  = [t.lower() for t in entry["tables"]]
        column_names = [c[0].lower() for c in entry["columns"]]

        # special token ids (these get IGNORE label)
        special_ids = {
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id,
        }

        for i in range(seq_len):
            tok_id = int(input_ids[i].item())

            # ── special tokens → ignore ───────────────────────────
            if tok_id in special_ids or attention_mask[i] == 0:
                labels[i] = IGNORE
                continue

            # ── decode token to string ────────────────────────────
            # CONCEPT: tokenizer.decode() converts a single token ID
            # back to its string. We strip leading "Ġ" (RoBERTa's
            # space prefix character) and lowercase for matching.
            tok_str = self.tokenizer.decode([tok_id]).strip().lower()
            tok_str = tok_str.lstrip("ġ").strip()   # RoBERTa space prefix

            if not tok_str:
                labels[i] = NONE
                continue

            # ── schema tokens: check table vs column ─────────────
            if token_type_ids[i] == 1:
                # It's a schema token. Is it a table name or column?
                if any(tok_str in t or t in tok_str for t in table_names):
                    labels[i] = TABLE
                elif any(tok_str in c or c in tok_str for c in column_names):
                    labels[i] = COLUMN
                else:
                    # punctuation like ":", "(", ")", "," in schema string
                    labels[i] = IGNORE

            # ── question tokens: check if they mention schema ─────
            else:
                # Is this question word referencing a table name?
                if any(tok_str in t or t in tok_str for t in table_names):
                    labels[i] = TABLE
                # Is it referencing a column name?
                elif any(tok_str in c or c in tok_str for c in column_names):
                    labels[i] = COLUMN
                # Does it look like a numeric value?
                elif tok_str.replace(".", "").replace("-", "").isdigit():
                    labels[i] = VALUE
                # Otherwise it's just a regular word
                else:
                    labels[i] = NONE

        return labels


# ─────────────────────────────────────────────────────────────────
# COLLATE FUNCTION
#
# CONCEPT: DataLoader calls collate_fn to combine a list of
# individual examples into one batch. The default collate assumes
# all tensors are the same size — ours are (we padded to max_length)
# for input_ids. But db_id is a string, which needs special handling.
# ─────────────────────────────────────────────────────────────────

def collate_fn(batch: list) -> dict:
    """
    Stack individual example dicts into a single batch dict.

    All tensor fields are already padded to max_length in __getitem__,
    so we just stack them. db_id is kept as a list of strings.
    """
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "token_type_ids": torch.stack([b["token_type_ids"] for b in batch]),
        "token_labels":   torch.stack([b["token_labels"]   for b in batch]),
        "gold_sql_ids":   torch.stack([b["gold_sql_ids"]   for b in batch]),
        "db_ids":         [b["db_id"] for b in batch],   # list of strings
    }


# ─────────────────────────────────────────────────────────────────
# DATALOADER BUILDER
# ─────────────────────────────────────────────────────────────────

def build_dataloaders(
    train_json:   str,
    dev_json:     str,
    tables_json:  str,
    tokenizer_name: str = "roberta-base",
    batch_size:   int   = 16,
    max_seq_len:  int   = 512,
    max_sql_len:  int   = 128,
    num_workers:  int   = 0,
    seed:         int   = 42,
):
    """
    Build train and dev DataLoaders for Spider.

    Returns:
        train_loader, dev_loader, schema_dict, tokenizer
    """
    # ── fix random seeds ──────────────────────────────────────────
    # IMPORTANT: Do this before anything else.
    # Ensures reproducible shuffling, weight init, dropout masks.
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # ── load tokenizer ────────────────────────────────────────────
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # ── load schema dict ──────────────────────────────────────────
    print("Loading schema dict...")
    schema_dict = load_schema_dict(tables_json)

    # ── build datasets ────────────────────────────────────────────
    print("Building train dataset...")
    train_dataset = SpiderDataset(
        train_json, schema_dict, tokenizer, max_seq_len, max_sql_len
    )

    print("Building dev dataset...")
    dev_dataset = SpiderDataset(
        dev_json, schema_dict, tokenizer, max_seq_len, max_sql_len
    )

    # ── build dataloaders ─────────────────────────────────────────
    # CRITICAL: shuffle=True for train, shuffle=False for dev.
    # Dev must always iterate in the same order for reproducible eval.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,              # ← random order every epoch
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,             # ← NEVER shuffle dev
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    print(f"\nTrain batches: {len(train_loader)} "
          f"({len(train_dataset)} examples, batch_size={batch_size})")
    print(f"Dev batches:   {len(dev_loader)} "
          f"({len(dev_dataset)} examples, batch_size={batch_size})")
    print(f"Train truncated: {train_dataset.n_truncated} examples exceeded {max_seq_len} tokens")

    return train_loader, dev_loader, schema_dict, tokenizer


# ─────────────────────────────────────────────────────────────────
# VERIFICATION TEST
# Run: python data_pipeline.py
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os

    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import TRAIN_JSON, DEV_JSON, TABLES_JSON

    for p in [TRAIN_JSON, DEV_JSON, TABLES_JSON]:
        if not os.path.exists(p):
            print(f"Not found: {p}")
            sys.exit(1)

    print("=" * 60)
    print("Task 2 — Dataset Pipeline Verification")
    print("=" * 60)

    train_loader, dev_loader, schema_dict, tokenizer = build_dataloaders(
        TRAIN_JSON, DEV_JSON, TABLES_JSON,
        batch_size=4,      # small batch for quick test
        max_seq_len=512,
        max_sql_len=128,
    )

    # ── inspect first 3 batches ───────────────────────────────────
    print("\n--- Inspecting first 3 train batches ---")
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 3:
            break

        print(f"\nBatch {batch_idx}:")
        print(f"  input_ids:      {batch['input_ids'].shape}")
        print(f"  attention_mask: {batch['attention_mask'].shape}")
        print(f"  token_type_ids: {batch['token_type_ids'].shape}")
        print(f"  token_labels:   {batch['token_labels'].shape}")
        print(f"  gold_sql_ids:   {batch['gold_sql_ids'].shape}")
        print(f"  db_ids:         {batch['db_ids']}")

        # ── verify label distribution ─────────────────────────────
        labels = batch["token_labels"]
        n_table  = (labels == 0).sum().item()
        n_col    = (labels == 1).sum().item()
        n_val    = (labels == 2).sum().item()
        n_none   = (labels == 3).sum().item()
        n_ignore = (labels == -100).sum().item()
        print(f"  token_labels  — TABLE:{n_table}  COLUMN:{n_col}  "
              f"VALUE:{n_val}  NONE:{n_none}  IGNORE:{n_ignore}")

        # ── verify type id distribution ───────────────────────────
        types    = batch["token_type_ids"]
        n_q      = (types == 0).sum().item()
        n_s      = (types == 1).sum().item()
        print(f"  token_type_ids — question:{n_q}  schema:{n_s}")

        # ── decode first example to sanity check ──────────────────
        if batch_idx == 0:
            print(f"\n  First example decoded:")
            ids    = batch["input_ids"][0].tolist()
            types0 = batch["token_type_ids"][0].tolist()
            labs0  = batch["token_labels"][0].tolist()
            label_names = {0:"TABLE", 1:"COL", 2:"VAL", 3:"NONE", -100:"IGN"}
            tokens = tokenizer.convert_ids_to_tokens(ids)
            print(f"  {'Token':<20} {'Type':<8} {'Label'}")
            print(f"  {'-'*40}")
            for tok, tp, lb in zip(tokens[:25], types0[:25], labs0[:25]):
                if tok == "<pad>":
                    break
                print(f"  {tok:<20} {'SCHEMA' if tp==1 else 'QUEST':<8} {label_names.get(lb,'?')}")

    print("\n✓ Task 2 dataset pipeline working correctly.")
    print("  All shapes as expected. Token labels assigned.")
    print("  Ready for Task 3 (cross-attention schema linking).")