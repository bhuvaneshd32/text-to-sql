"""
eval_utils.py
-------------
Shared evaluation utilities for Text-to-SQL.
"""

import os
import sys
import re
import sqlite3
import json
import argparse
from typing import List

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────
# SQL NORMALISATION
# ─────────────────────────────────────────────────────────────────

def normalise_sql(sql: str) -> str:
    """
    Normalise SQL string before comparison.

    
    
    Examples:
        "SELECT COUNT ( * ) FROM employees"
        → "select count(*) from employees"

        "SELECT   name  ,  salary  FROM   employees"
        → "select name, salary from employees"
    """
    if not sql or not sql.strip():
        return ""

    sql = sql.strip()

    # 1. lowercase
    sql = sql.lower()

    # 2. normalise quotes — replace " with '
    sql = sql.replace('"', "'")

    # 3. remove spaces inside parentheses: count ( * ) → count(*)
    sql = re.sub(r'\(\s+', '(', sql)
    sql = re.sub(r'\s+\)', ')', sql)

    # 4. normalise spaces around commas: a , b → a, b
    sql = re.sub(r'\s*,\s*', ', ', sql)

    # 5. normalise spaces around = != > < >= <=
    sql = re.sub(r'\s*(=|!=|<>|>=|<=|>|<)\s*', r' \1 ', sql)

    # 6. collapse all whitespace sequences to single space
    sql = re.sub(r'\s+', ' ', sql)

    # 7. strip again after all replacements
    sql = sql.strip()

    return sql


# ─────────────────────────────────────────────────────────────────
# CORE METRICS
# ─────────────────────────────────────────────────────────────────

def execute_sql(sql: str, db_path: str):
    """
    Execute SQL on a SQLite database and return normalised result rows.
    Returns None if execution fails.
    """
    try:
        conn   = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows   = cursor.fetchall()
        conn.close()
        # normalise result values: lowercase, strip whitespace
        # sort rows so order does not affect comparison
        return sorted([
            tuple(str(v).lower().strip() for v in row)
            for row in rows
        ])
    except Exception:
        return None


def exec_accuracy(pred_sql: str, gold_sql: str, db_path: str) -> int:
    
    pred_norm = normalise_sql(pred_sql)
    gold_norm = normalise_sql(gold_sql)

    pred_result = execute_sql(pred_norm, db_path)
    gold_result = execute_sql(gold_norm, db_path)

    if pred_result is None or gold_result is None:
        return 0

    return int(pred_result == gold_result)


def result_set_f1(pred_sql: str, gold_sql: str, db_path: str) -> float:
    """
    Result-Set F1 — F1 between result sets treating rows as tokens.
    Gives partial credit for near-correct queries.
    """
    pred_norm = normalise_sql(pred_sql)
    gold_norm = normalise_sql(gold_sql)

    pred_result = execute_sql(pred_norm, db_path)
    gold_result = execute_sql(gold_norm, db_path)

    if pred_result is None or gold_result is None:
        return 0.0

    if len(pred_result) == 0 and len(gold_result) == 0:
        return 1.0

    pred_set = set(pred_result)
    gold_set = set(gold_result)

    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0

    tp        = len(pred_set & gold_set)
    precision = tp / len(pred_set)
    recall    = tp / len(gold_set)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def string_match_accuracy(pred_sql: str, gold_sql: str) -> int:
    return int(normalise_sql(pred_sql) == normalise_sql(gold_sql))

# GREEDY DECODE (module-level so show_samples can call it)
def greedy_decode(model, batch, tokenizer, max_len=128, device="cpu"):
    """
    Autoregressive greedy decode — feeds model's own output back in.
    Used during evaluation only.
    """
    model.eval()
    with torch.no_grad():
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)

        all_hidden = model.encode(input_ids, attention_mask, token_type_ids)
        batch_size = input_ids.shape[0]

        bos_id    = tokenizer.bos_token_id or 0
        eos_id    = tokenizer.eos_token_id or 2
        dec_input = torch.full(
            (batch_size, 1), bos_id, dtype=torch.long, device=device
        )

        generated = [[] for _ in range(batch_size)]
        done      = [False] * batch_size

        for _ in range(max_len):
            sql_len   = dec_input.shape[1]
            positions = torch.arange(sql_len, device=device).unsqueeze(0)
            dec_emb   = (
                model.token_embedding(dec_input)
                + model.pos_embedding(positions)
            )
            causal_mask  = nn.Transformer.generate_square_subsequent_mask(
                sql_len, device=device
            )
            enc_pad_mask = (attention_mask == 0)
            dec_hidden   = model.decoder(
                tgt=dec_emb, memory=all_hidden,
                tgt_mask=causal_mask,
                memory_key_padding_mask=enc_pad_mask,
            )
            logits     = model.output_projection(dec_hidden[:, -1, :])
            next_token = logits.argmax(dim=-1, keepdim=True)

            for b in range(batch_size):
                if not done[b]:
                    tid = next_token[b, 0].item()
                    if tid == eos_id:
                        done[b] = True
                    else:
                        generated[b].append(tid)

            if all(done):
                break
            dec_input = torch.cat([dec_input, next_token], dim=1)

    return [
        tokenizer.decode(g, skip_special_tokens=True).strip()
        for g in generated
    ]


def evaluate_checkpoint(
    checkpoint_path: str,
    dev_json:        str,
    tables_json:     str,
    db_dir:          str,
    batch_size:      int  = 8,
    max_batches:     int  = None,
    use_normalised_string: bool = True,
):
    """
    Run full evaluation on Spider dev set using the saved checkpoint.
    Reports EX and F1 broken down by difficulty level.

    Args:
        use_normalised_string: if True, also reports normalised string match
                               alongside execution accuracy.
    """
    from transformers import AutoTokenizer
    from nlp.data_pipeline import build_dataloaders
    from nlp.multi_task    import TextToSQLModel

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("t5-large")

    print(f"Loading checkpoint: {checkpoint_path}", flush=True)
    model = TextToSQLModel.load_for_rl(checkpoint_path)
    model = model.to(device)
    model.eval()

    _, dev_loader, schema_dict, _ = build_dataloaders(
        dev_json, dev_json, tables_json,
        batch_size=batch_size,
    )

    with open(dev_json) as f:
        dev_examples = json.load(f)

    results = {
        "easy":       {"ex": [], "f1": [], "sm": []},
        "medium":     {"ex": [], "f1": [], "sm": []},
        "hard":       {"ex": [], "f1": [], "sm": []},
        "extra hard": {"ex": [], "f1": [], "sm": []},
        "all":        {"ex": [], "f1": [], "sm": []},
    }

    example_idx = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dev_loader):
            if max_batches and batch_idx >= max_batches:
                break

            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            bs             = input_ids.shape[0]

            # beam search decode
            generated = model.generate_sql(
                input_ids, attention_mask,
                max_length=128,
                num_beams=4,
            )
            pred_sqls_batch = tokenizer.batch_decode(generated, skip_special_tokens=True)

            for b in range(bs):
                if example_idx >= len(dev_examples):
                    break

                pred_sql = pred_sqls_batch[b].strip()
                ex_info  = dev_examples[example_idx]
                gold_sql = ex_info["query"]
                db_id    = ex_info["db_id"]
                diff     = ex_info.get("difficulty", "all").lower()
                db_path  = os.path.join(db_dir, db_id, f"{db_id}.sqlite")

                ex_score = exec_accuracy(pred_sql, gold_sql, db_path)
                f1_score = result_set_f1(pred_sql, gold_sql, db_path)
                sm_score = string_match_accuracy(pred_sql, gold_sql)

                results["all"]["ex"].append(ex_score)
                results["all"]["f1"].append(f1_score)
                results["all"]["sm"].append(sm_score)
                if diff in results:
                    results[diff]["ex"].append(ex_score)
                    results[diff]["f1"].append(f1_score)
                    results[diff]["sm"].append(sm_score)

                example_idx += 1

            if batch_idx % 10 == 0:
                n = len(results["all"]["ex"])
                if n > 0:
                    running_ex = sum(results["all"]["ex"]) / n
                    running_sm = sum(results["all"]["sm"]) / n
                    print(
                        f"  [{batch_idx}] {n} examples — "
                        f"EX: {running_ex*100:.1f}%  "
                        f"String match: {running_sm*100:.1f}%",
                        flush=True,
                    )

    # ── results table ─────────────────────────────────────────────
    print(f"\n{'='*70}", flush=True)
    print(f"Evaluation Results — {checkpoint_path}", flush=True)
    print(f"{'='*70}", flush=True)
    print(
        f"{'Difficulty':<15} {'Count':>6} {'EX%':>8} {'F1%':>8} {'String Match%':>15}",
        flush=True,
    )
    print(f"{'-'*55}", flush=True)

    eval_results = {}
    for diff in ["easy", "medium", "hard", "extra hard", "all"]:
        exs = results[diff]["ex"]
        f1s = results[diff]["f1"]
        sms = results[diff]["sm"]
        if not exs:
            continue
        avg_ex = sum(exs) / len(exs)
        avg_f1 = sum(f1s) / len(f1s)
        avg_sm = sum(sms) / len(sms)
        eval_results[diff] = {
            "ex": avg_ex, "f1": avg_f1, "string_match": avg_sm, "count": len(exs)
        }
        print(
            f"{diff:<15} {len(exs):>6} "
            f"{avg_ex*100:>7.1f}% "
            f"{avg_f1*100:>7.1f}% "
            f"{avg_sm*100:>14.1f}%",
            flush=True,
        )

    print(f"{'='*70}", flush=True)

    # explanation of each metric
    print(f"\nMetric definitions:", flush=True)
    print(f"  EX%           — both SQLs executed on SQLite; result sets compared", flush=True)
    print(f"  F1%           — F1 between result sets; partial credit for near-misses", flush=True)
    print(f"  String match% — normalised SQL strings compared (case + whitespace insensitive)", flush=True)

    with open("eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nSaved to eval_results.json", flush=True)

    return eval_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Text-to-SQL checkpoint")
    parser.add_argument("--checkpoint",   default="checkpoints/pretrained_best.pt")
    parser.add_argument("--max_batches",  type=int,  default=None)
    parser.add_argument("--batch_size",   type=int,  default=8)
    parser.add_argument("--show_samples", action="store_true",
                        help="Print sample predictions vs gold SQL")
    args = parser.parse_args()

    from config import DEV_JSON, TABLES_JSON, DB_DIR, TRAIN_JSON

    # ── full evaluation ───────────────────────────────────────────
    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        dev_json=DEV_JSON,
        tables_json=TABLES_JSON,
        db_dir=DB_DIR,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
    )

    # ── sample predictions ────────────────────────────────────────
    if args.show_samples:
        from transformers import AutoTokenizer
        from nlp.data_pipeline import build_dataloaders
        from nlp.multi_task    import TextToSQLModel

        print("\n" + "="*60, flush=True)
        print("Sample Predictions", flush=True)
        print("="*60, flush=True)

        device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # UPGRADED TO T5-LARGE
        tokenizer = AutoTokenizer.from_pretrained("t5-large")
        model     = TextToSQLModel.load_for_rl(args.checkpoint).to(device)
        model.eval()

        _, dev_loader, _, _ = build_dataloaders(
            TRAIN_JSON, DEV_JSON, TABLES_JSON, batch_size=8
        )

        with open(DEV_JSON) as f:
            dev_examples = json.load(f)

        batch = next(iter(dev_loader))
        preds = greedy_decode(model, batch, tokenizer, device=device)

        for i, pred in enumerate(preds[:8]):
            if i >= len(dev_examples):
                break
            ex       = dev_examples[i]
            gold     = ex["query"]
            ex_sc    = 1 if normalise_sql(pred) == normalise_sql(gold) else 0
            match_ch = "✓" if ex_sc else "✗"

            print(f"\n[{i+1}] {match_ch}", flush=True)
            print(f"  Question: {ex['question']}", flush=True)
            print(f"  Gold:     {gold}", flush=True)
            print(f"  Pred:     {pred}", flush=True)
            print(f"  Gold norm: {normalise_sql(gold)}", flush=True)
            print(f"  Pred norm: {normalise_sql(pred)}", flush=True)