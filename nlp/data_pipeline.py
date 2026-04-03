"""
data_pipeline.py
----------------
Spider dataset pipeline for T5-based Text-to-SQL.

INPUT FORMAT (T5 prompt):
    "translate to SQL: How many employees earn above 5000?
     tables: employees ( id INT , name TEXT , salary REAL )"

OUTPUT (label):
    "SELECT count(*) FROM employees WHERE salary > 5000"

T5's tokenizer handles BOS/EOS automatically — no manual addition needed.
"""

import json
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from nlp.schema_utils import load_schema_dict
from config import T5_MODEL
# T5 tokenizer
T5_TOKENIZER = T5_MODEL

# label constants — kept for cross_attention compatibility
TABLE  = 0
COLUMN = 1
VALUE  = 2
NONE   = 3
IGNORE = -100


def format_t5_input(question: str, db_id: str, schema_dict: dict,
                    max_length: int = 512) -> str:
    """
    PICARD-style schema format — proven to work well with T5 on Spider.
    Format:
        "question: How many perpetrators? | col : Perpetrator_ID | col : Killed | 
         table : perpetrator | col : People_ID | col : Name | table : people"
    """
    entry  = schema_dict[db_id]
    tables = entry["tables"]
    cols   = entry["columns"]

    # group columns by table, preserve order
    table_cols = {t: [] for t in tables}
    for col_name, col_type, t_idx in cols:
        table_cols[tables[t_idx]].append(col_name)

    parts = [f"question: {question}"]
    for t in tables:
        for col in table_cols[t]:
            parts.append(f"col : {col}")
        parts.append(f"table : {t}")

    return " | ".join(parts)


class SpiderDataset(Dataset):
    """Spider dataset for T5 fine-tuning."""

    def __init__(
        self,
        data_json_path: str,
        schema_dict:    dict,
        tokenizer,
        max_input_len:  int = 512,
        max_target_len: int = 128,
    ):
        self.schema_dict    = schema_dict
        self.tokenizer      = tokenizer
        self.max_input_len  = max_input_len
        self.max_target_len = max_target_len

        with open(data_json_path) as f:
            raw = json.load(f)

        self.examples = [ex for ex in raw if ex["db_id"] in schema_dict]
        n_dropped = len(raw) - len(self.examples)
        if n_dropped > 0:
            print(f"[WARNING] Dropped {n_dropped} examples", flush=True)

        print(f"Loaded {len(self.examples)} examples from "
              f"{os.path.basename(data_json_path)}", flush=True)
        self.n_truncated = 0

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex       = self.examples[idx]
        question = ex["question"]
        gold_sql = ex["query"]
        db_id    = ex["db_id"]

        # ── format input prompt ───────────────────────────────────
        prompt = format_t5_input(question, db_id, self.schema_dict)

        # ── tokenize input ────────────────────────────────────────
        enc = self.tokenizer(
            prompt,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        if int(attention_mask.sum()) == self.max_input_len:
            self.n_truncated += 1

        # ── tokenize target (gold SQL) ────────────────────────────
        # Use text_target instead of the deprecated as_target_tokenizer()
        tgt = self.tokenizer(
            text_target=gold_sql,
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = tgt["input_ids"].squeeze(0)
        # replace pad token id with -100 so loss ignores padding
        labels[labels == self.tokenizer.pad_token_id] = -100

        # ── dummy tensors for API compatibility ───────────────────
        seq_len        = input_ids.shape[0]
        token_type_ids = torch.zeros(seq_len, dtype=torch.long)
        token_labels   = torch.full((seq_len,), IGNORE, dtype=torch.long)

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,   # unused by T5
            "token_labels":   token_labels,     # unused by T5
            "gold_sql_ids":   labels,
            "db_id":          db_id,
            "question":       question,
            "gold_sql":       gold_sql,
        }


def collate_fn(batch: list) -> dict:
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "token_type_ids": torch.stack([b["token_type_ids"] for b in batch]),
        "token_labels":   torch.stack([b["token_labels"]   for b in batch]),
        "gold_sql_ids":   torch.stack([b["gold_sql_ids"]   for b in batch]),
        "db_ids":         [b["db_id"]    for b in batch],
        "questions":      [b["question"] for b in batch],
        "gold_sqls":      [b["gold_sql"] for b in batch],
    }


def build_dataloaders(
    train_json:     str,
    dev_json:       str,
    tables_json:    str,
    tokenizer_name: str = T5_TOKENIZER,
    batch_size:     int = 16,
    max_seq_len:    int = 512,
    max_sql_len:    int = 128,
    num_workers:    int = 0,
    seed:           int = 42,
):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    print(f"Loading tokenizer: {tokenizer_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    print("Loading schema dict...", flush=True)
    schema_dict = load_schema_dict(tables_json)

    print("Building train dataset...", flush=True)
    train_dataset = SpiderDataset(
        train_json, schema_dict, tokenizer, max_seq_len, max_sql_len
    )
    print("Building dev dataset...", flush=True)
    dev_dataset = SpiderDataset(
        dev_json, schema_dict, tokenizer, max_seq_len, max_sql_len
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, collate_fn=collate_fn, num_workers=num_workers,
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=num_workers,
    )

    print(f"\nTrain: {len(train_loader)} batches ({len(train_dataset)} examples)", flush=True)
    print(f"Dev:   {len(dev_loader)} batches ({len(dev_dataset)} examples)", flush=True)
    print(f"Truncated: {train_dataset.n_truncated}", flush=True)

    return train_loader, dev_loader, schema_dict, tokenizer