import os, sys, json, time, random, argparse
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TRAIN_JSON, DEV_JSON, TABLES_JSON, DB_DIR,
    CHECKPOINT_DIR, BEST_CHECKPOINT, LAST_CHECKPOINT,
    RANDOM_SEED,
)
from data_pipeline import build_dataloaders
from multi_task    import TextToSQLModel

try:
    from tqdm import tqdm
    TQDM = True
except ImportError:
    TQDM = False

try:
    import wandb
    WANDB = True
except ImportError:
    WANDB = False


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_exact_match(model, dev_loader, tokenizer, device, max_batches=50):
    """
    Normalised string match using beam search decode.
    Much more accurate than greedy decode for T5.
    """
    import re

    def norm(sql):
        sql = sql.lower().strip()
        sql = re.sub(r'\s+', ' ', sql)
        sql = re.sub(r'\(\s+', '(', sql)
        sql = re.sub(r'\s+\)', ')', sql)
        return sql

    model.eval()
    total = correct = 0

    iterator = dev_loader
    if TQDM:
        iterator = tqdm(dev_loader, desc="  Eval", leave=False, ncols=80)

    with torch.no_grad():
        for batch_idx, batch in enumerate(iterator):
            if max_batches and batch_idx >= max_batches:
                break

            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            gold_sqls      = batch["gold_sqls"]

            # FIX 4: beam search generation — better quality than greedy
            generated = model.generate_sql(
                input_ids, attention_mask,
                max_length=128,
                num_beams=4,
                length_penalty=0.6,      # Penalises overly long SQL
                no_repeat_ngram_size=3,  # Prevents repetition loops
            )

            pred_sqls = tokenizer.batch_decode(generated, skip_special_tokens=True)

            for pred, gold in zip(pred_sqls, gold_sqls):
                total   += 1
                correct += int(norm(pred) == norm(gold))

    return correct / total if total > 0 else 0.0


def train(args):
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}", flush=True)
    print(f"Text-to-SQL T5 Fine-Tuning", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Device:     {device}", flush=True)
    print(f"Epochs:     {args.epochs}", flush=True)
    print(f"Batch size: {args.batch_size}", flush=True)
    print(f"Grad Accum: {args.grad_accum}", flush=True)
    print(f"LR:         {args.lr}", flush=True)
    print(f"{'='*60}\n", flush=True)

    train_loader, dev_loader, schema_dict, tokenizer = build_dataloaders(
        TRAIN_JSON, DEV_JSON, TABLES_JSON,
        tokenizer_name="t5-large",
        batch_size=args.batch_size,
        max_seq_len=512,
        max_sql_len=128,
    )

    print("Building model...", flush=True)
    model = TextToSQLModel().to(device)

    # FIX 1: Cosine decay with warmup and AdamW eps
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        eps=1e-8,
    )
    
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=500
    )
    # Adjust total steps for CosineAnnealingLR considering gradient accumulation
    total_steps = (args.epochs * len(train_loader)) // args.grad_accum
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - 500), eta_min=1e-5
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[500]
    )

    if WANDB and args.use_wandb:
        wandb.init(project="text-to-sql-t5", config=vars(args))

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    best_dev_ex  = 0.0
    global_step  = 0
    training_log = []

    # resume
    start_epoch = 1
    if args.resume and os.path.exists(LAST_CHECKPOINT):
        ckpt = torch.load(LAST_CHECKPOINT, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_dev_ex = ckpt.get("dev_ex", 0.0)
        print(f"Resumed from epoch {start_epoch-1}, best_dev_ex={best_dev_ex*100:.2f}%", flush=True)

    print(f"\nStarting training — {len(train_loader)} batches/epoch\n", flush=True)

    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_n    = 0
        start      = time.time()
        
        optimizer.zero_grad() # Ensure clean slate at the start of the epoch

        iterator = tqdm(train_loader, desc=f"Epoch {epoch:2d}", ncols=100) if TQDM else train_loader

        for batch_idx, batch in enumerate(iterator):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["gold_sql_ids"].to(device)

            # T5 forward pass
            loss, L_sql, L_schema, _, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            # FIX 5: Gradient Accumulation
            loss = loss / args.grad_accum
            loss.backward()

            if (batch_idx + 1) % args.grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            # Multiply back to log the true loss magnitude
            epoch_loss += loss.item() * args.grad_accum
            epoch_n    += 1

            # heartbeat first 5 batches
            if epoch == start_epoch and batch_idx < 5:
                print(f"  [batch {batch_idx+1}/5]  loss={(loss.item() * args.grad_accum):.4f}", flush=True)

            if TQDM and batch_idx % 10 == 0:
                iterator.set_postfix({"loss": f"{epoch_loss/epoch_n:.4f}"})
            elif not TQDM and global_step % 50 == 0 and (batch_idx + 1) % args.grad_accum == 0:
                print(f"Epoch {epoch} | Step {global_step} | loss={epoch_loss/epoch_n:.4f} | lr={scheduler.get_last_lr()[0]:.2e}", flush=True)

            if WANDB and args.use_wandb and global_step % 50 == 0 and (batch_idx + 1) % args.grad_accum == 0:
                wandb.log({"train/loss": epoch_loss/epoch_n, "step": global_step})

        # ── epoch end ──────────────────────────────────────────────
        elapsed    = time.time() - start
        avg_loss   = epoch_loss / max(epoch_n, 1)

        print(f"\n{'─'*60}", flush=True)
        print(f"Epoch {epoch} done  ({elapsed:.0f}s)  loss={avg_loss:.4f}", flush=True)

        # evaluate
        print(f"  Evaluating ({args.eval_batches} batches)...", flush=True)
        dev_ex = evaluate_exact_match(
            model, dev_loader, tokenizer, device, max_batches=args.eval_batches
        )
        print(f"  Dev string match: {dev_ex*100:.2f}%", flush=True)

        # log
        entry = {"epoch": epoch, "step": global_step, "train_loss": avg_loss, "dev_ex": dev_ex}
        training_log.append(entry)
        with open("training_log.json", "w") as f:
            json.dump(training_log, f, indent=2)

        if WANDB and args.use_wandb:
            wandb.log({"epoch/dev_ex": dev_ex, "epoch/train_loss": avg_loss, "epoch": epoch})

        # save best
        if dev_ex > best_dev_ex:
            best_dev_ex = dev_ex
            model.save_checkpoint(
                BEST_CHECKPOINT,
                extra={
                    "config": {
                        "t5_model": "t5-large", "max_sql_len": 128,
                        "vocab_size": model.vocab_size,
                        "decoder_layers": 0, "decoder_ff_dim": 0, "decoder_heads": 0,
                    },
                    "dev_ex": dev_ex, "dev_f1": 0.0, "epoch": epoch,
                }
            )
            print(f"  ✓ Best checkpoint saved (dev_ex={dev_ex*100:.2f}%)", flush=True)

        model.save_checkpoint(
            LAST_CHECKPOINT,
            extra={"config": {"t5_model": "t5-large"}, "epoch": epoch, "dev_ex": dev_ex}
        )

        print(f"{'─'*60}\n", flush=True)

    print(f"{'='*60}", flush=True)
    print(f"Training complete.  Best dev match: {best_dev_ex*100:.2f}%", flush=True)
    if WANDB and args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",       type=int,   default=15)
    parser.add_argument("--batch_size",   type=int,   default=16)
    parser.add_argument("--grad_accum",   type=int,   default=2)    # ADDED
    parser.add_argument("--lr",           type=float, default=3e-4) # Modified
    parser.add_argument("--eval_batches", type=int,   default=50)
    parser.add_argument("--resume",       action="store_true")
    parser.add_argument("--use_wandb",    action="store_true")
    args = parser.parse_args()
    train(args)