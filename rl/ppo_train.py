"""
rl/ppo_train.py
---------------
PPO fine-tuning for Text-to-SQL using T5 backbone.

RUN:
    python -m rl.ppo_train --episodes 50 --batch_episodes 8   # sanity check
    python -m rl.ppo_train --episodes 3000 --use_wandb        # full training
"""

import os
import sys
import json
import re
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.distributions import Categorical
from transformers.modeling_outputs import BaseModelOutput
from rl.reward import compute_reward

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    TRAIN_JSON, DEV_JSON, TABLES_JSON, DB_DIR,
    CHECKPOINT_DIR, BEST_CHECKPOINT,
)
from nlp.multi_task    import TextToSQLModel
from nlp.data_pipeline import build_dataloaders
from nlp.schema_utils  import load_schema_dict
from nlp.eval_utils    import exec_accuracy, result_set_f1
from rl.environment    import TextToSQLEnv

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────
# VALUE HEAD
# ─────────────────────────────────────────────────────────────────

class ValueHead(nn.Module):
    """Estimates V(s) from decoder's last hidden state [batch, d_model]."""
    def __init__(self, hidden_size: int = 1024):  # 768 for t5-base, 1024 for t5-large
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden).squeeze(-1)


# ─────────────────────────────────────────────────────────────────
# DECODE STEP
# ─────────────────────────────────────────────────────────────────

def decode_step_with_hidden(model, memory, generated_tokens, device):
    """
    One-step T5 decode. Returns logits [vocab] and last decoder hidden [1, 768].

    memory: encoder last_hidden_state [1, seq_len, 768]
    We pass it as encoder_outputs to avoid re-encoding every step.
    """
    t5    = model.t5
    bos   = t5.config.decoder_start_token_id or t5.config.pad_token_id

    if len(generated_tokens) == 0:
        dec_ids = torch.tensor([[bos]], device=device)
    else:
        dec_ids = torch.tensor([generated_tokens], device=device)

    encoder_outputs = BaseModelOutput(last_hidden_state=memory)

    outputs = t5(
        encoder_outputs=encoder_outputs,
        decoder_input_ids=dec_ids,
        output_hidden_states=True,
    )

    logits      = outputs.logits[0, -1, :]                      # [vocab]
    last_hidden = outputs.decoder_hidden_states[-1][:, -1, :]   # [1, 768]
    return logits, last_hidden


# ─────────────────────────────────────────────────────────────────
# GAE
# ─────────────────────────────────────────────────────────────────

def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    T          = len(rewards)
    advantages = torch.zeros(T)
    gae        = 0.0
    next_value = 0.0

    for t in reversed(range(T)):
        delta         = rewards[t] + gamma * next_value - values[t]
        gae           = delta + gamma * lam * gae
        advantages[t] = gae
        next_value    = values[t]

    returns = advantages + torch.tensor(values)

    if advantages.numel() > 1 and advantages.std() > 1e-8:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns


# ─────────────────────────────────────────────────────────────────
# COLLECT TRAJECTORY
# ─────────────────────────────────────────────────────────────────

def normalize_sql(sql):
    sql = re.sub(r"\s+", " ", sql)
    sql = sql.replace("(", " ( ").replace(")", " ) ")
    sql = sql.replace(",", " , ")
    return sql.lower().strip()


def collect_trajectory(env, model, value_head, device, temperature=0.8):
    state  = env.reset()
    memory = env.current_memory.to(device)

    actions, log_probs, values, rewards, hiddens = [], [], [], [], []
    done     = False
    max_steps = 64

    while not done and len(actions) < max_steps:
        with torch.no_grad():
            logits, last_hidden = decode_step_with_hidden(
                model, memory, env.generated_tokens, device
            )
            value = value_head(last_hidden).item()

        partial    = normalize_sql(env.decode_sql(env.generated_tokens))
        mask       = env.fsm.get_mask(partial)
        logits     = logits.float()
        vocab_size = logits.shape[0]

        # ── 1. pad/trim mask to match T5 vocab FIRST ──────────────
        if mask.shape[0] < vocab_size:
            pad  = torch.zeros(vocab_size - mask.shape[0], dtype=torch.bool)
            mask = torch.cat([mask, pad])
        elif mask.shape[0] > vocab_size:
            mask = mask[:vocab_size]

        # ── 2. now safe to index logits with mask ─────────────────
        valid_count = mask.sum().item()
        if valid_count < 5:
            valid_logits = logits.clone()
            valid_logits[~mask] = -1e9
            top_valid    = valid_logits.topk(min(5, int(valid_count))).indices
            mask         = mask.clone()
            mask[top_valid] = True

        logits[~mask] = -1e9

        probs = F.softmax(logits / temperature, dim=-1)
        probs = probs.clamp(min=1e-10)
        probs = probs / probs.sum()

        if torch.isnan(probs).any() or probs.sum() < 1e-9:
            action = logits.argmax().item()
            lp     = F.log_softmax(logits, dim=-1)[action].item()
        else:
            dist   = Categorical(probs)
            action = dist.sample().item()
            lp     = dist.log_prob(torch.tensor(action, device=device)).item()

        _, reward, done, _ = env.step(action)

        actions.append(action)
        log_probs.append(lp)
        values.append(value)
        rewards.append(reward)
        hiddens.append(last_hidden.squeeze(0).detach().cpu())

    if rewards and rewards[-1] <= -0.9:
        print(f"[PENALTY] Episode ended with penalty at step {len(actions)}", flush=True)

    return {
        "actions":      actions,
        "log_probs":    log_probs,
        "values":       values,
        "rewards":      rewards,
        "hiddens":      hiddens,
        "steps":        len(actions),
        "final_reward": rewards[-1] if rewards else 0.0,
    }


# ─────────────────────────────────────────────────────────────────
# PPO UPDATE
# ─────────────────────────────────────────────────────────────────

def ppo_update(
    model, value_head, ref_model, optimizer, trajectories, device,
    clip_eps=0.2, vf_coef=0.5, ent_coef=0.01, kl_coef=0.1, ppo_epochs=4,
):
    all_actions, all_old_lps    = [], []
    all_advantages, all_returns = [], []
    all_hiddens                 = []

    for traj in trajectories:
        adv, ret = compute_gae(traj["rewards"], traj["values"])
        all_actions.extend(traj["actions"])
        all_old_lps.extend(traj["log_probs"])
        all_advantages.extend(adv.tolist())
        all_returns.extend(ret.tolist())
        all_hiddens.extend(traj["hiddens"])

    actions    = torch.tensor(all_actions,    device=device, dtype=torch.long)
    old_lps    = torch.tensor(all_old_lps,    device=device, dtype=torch.float32)
    advantages = torch.tensor(all_advantages, device=device, dtype=torch.float32)
    returns    = torch.tensor(all_returns,    device=device, dtype=torch.float32)
    hiddens = torch.stack(all_hiddens).to(device)   # [T, d_model]  

    loss_log = []

    for _ in range(ppo_epochs):
        # recompute logits from stored hidden states via lm_head
        new_logits    = model.t5.lm_head(hiddens)
        new_log_probs = F.log_softmax(new_logits, dim=-1)
        new_lps       = new_log_probs.gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        ratio    = torch.exp(new_lps - old_lps)
        surr1    = ratio * advantages
        surr2    = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        L_policy = -torch.min(surr1, surr2).mean()

        pred_values = value_head(hiddens).squeeze(-1)
        L_value     = F.mse_loss(pred_values, returns)

        probs   = F.softmax(new_logits, dim=-1)
        entropy = -(probs * (new_log_probs + 1e-8)).sum(dim=-1).mean()

        with torch.no_grad():
            ref_logits    = ref_model.t5.lm_head(hiddens)

        with torch.no_grad():
            ref_logits    = ref_model.t5.lm_head(hiddens)
        kl_div = F.kl_div(
            F.log_softmax(new_logits, dim=-1),
            F.softmax(ref_logits, dim=-1),
            reduction="batchmean",
        ).clamp(0, 1.0)

        loss = L_policy + vf_coef * L_value - ent_coef * entropy + kl_coef * kl_div

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(value_head.parameters()),
            max_norm=1.0,
        )
        optimizer.step()

        loss_log.append({
            "loss":    loss.item(),
            "policy":  L_policy.item(),
            "value":   L_value.item(),
            "entropy": entropy.item(),
            "kl":      kl_div.item(),
        })

    return {k: float(np.mean([d[k] for d in loss_log])) for k in loss_log[0]}


# ─────────────────────────────────────────────────────────────────
# EVALUATION — uses RL policy's token-by-token generation
# NOT beam search — beam search measures SL quality, not RL policy
# ─────────────────────────────────────────────────────────────────

def evaluate_rl(model, dev_loader, tokenizer, device, db_dir, n_batches=None):
    model.eval()
    ex_scores, f1_scores = [], []

    with torch.no_grad():
        for i, batch in enumerate(dev_loader):
            # if i >= n_batches:
            if n_batches is not None and i >= n_batches:    
                break

            input_ids      = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            gold_sql_ids   = batch["gold_sql_ids"][0]
            db_id          = batch["db_ids"][0]

            # greedy decode (temperature=0, deterministic) — best single sequence
            generated = model.generate_sql(
                input_ids.to(device),
                attention_mask.to(device),
                max_length=128,
                num_beams=4,
            )


            pred_sql = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
            gold_sql = batch["gold_sqls"][0] if "gold_sqls" in batch else \
           tokenizer.decode([t for t in gold_sql_ids.tolist() if t >= 0],
                            skip_special_tokens=True).strip()
            db_path  = os.path.join(db_dir, db_id, f"{db_id}.sqlite")

            ex_scores.append(exec_accuracy(pred_sql, gold_sql, db_path))
            f1_scores.append(result_set_f1(pred_sql, gold_sql, db_path))

    model.train()
    return {
        "exec_acc": float(np.mean(ex_scores)) if ex_scores else 0.0,
        "f1":       float(np.mean(f1_scores)) if f1_scores else 0.0,
    }


# ─────────────────────────────────────────────────────────────────
# CURRICULUM TIER
# ─────────────────────────────────────────────────────────────────

def get_curriculum_tier(episode: int) -> str:
    if episode >= 1000:
        return "nested"
    if episode >= 500:
        return "join"
    if episode >= 200:
        return "where"
    return "simple"


# ─────────────────────────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────────

def train_ppo(args):
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}")

    train_loader, dev_loader, _, tokenizer = build_dataloaders(
        TRAIN_JSON, DEV_JSON, TABLES_JSON, batch_size=1
    )

    # policy model — on device for gradient updates
    model = TextToSQLModel.load_for_rl(BEST_CHECKPOINT)
    model.to(device)
    print("=== BASELINE EXECUTION ACCURACY CHECK ===")
    if args.eval_only:
        print("=== EVAL ONLY MODE — no training ===", flush=True)
        eval_scores = evaluate_rl(
            model, dev_loader, tokenizer, device,
            db_dir=DB_DIR, n_batches=None,   # full dev set — fair comparison
        )
        print(f"exec_acc: {eval_scores['exec_acc']*100:.2f}%", flush=True)
        print(f"f1:       {eval_scores['f1']*100:.2f}%", flush=True)
        return

    model.train()
    for p in model.t5.encoder.parameters():
        p.requires_grad = False

    # ref model — stays on CPU for KL (avoids MPS sampling bug)
    ref_model = TextToSQLModel.load_for_rl(BEST_CHECKPOINT)
    ref_model.cpu()
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    optimizer = AdamW(
        list(model.t5.decoder.parameters()) +
        list(model.t5.lm_head.parameters()),
        lr=args.lr, weight_decay=0.01
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_exec_acc = 0.0
    log           = []
    data_iter     = iter(train_loader)
    reward_baseline = 0.0

    print("Starting REINFORCE training...\n")

    for episode in range(1, args.episodes + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        gold_sql_ids   = batch["gold_sql_ids"][0]
        db_id          = batch["db_ids"][0]
        gold_ids       = [t for t in gold_sql_ids.tolist() if t >= 0]
        gold_sql       = tokenizer.decode(gold_ids, skip_special_tokens=True).strip()
        db_path        = os.path.join(DB_DIR, db_id, f"{db_id}.sqlite")

        # ── SAMPLE on GPU ────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            sample_out = model.t5.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                do_sample=True,
                temperature=args.temperature,
                return_dict_in_generate=True,
                output_scores=True,
            )
        model.train()

        sequences = sample_out.sequences
        pred_ids  = sequences[0][1:]
        pred_sql  = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()

        # ── REWARD ───────────────────────────────────────────────
        sql_tokens = tokenizer.encode(pred_sql, add_special_tokens=False)
        reward = compute_reward(
            pred_sql=pred_sql,
            gold_sql=gold_sql,
            db_path=db_path,
            sql_tokens=sql_tokens,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            delta=args.delta,
        )
        reward_baseline = 0.9 * reward_baseline + 0.1 * reward
        advantage       = reward - reward_baseline

        if episode <= 30:
            print(f"[SAMPLE ep{episode}] reward={reward:.2f} | pred='{pred_sql[:60]}'", flush=True)

        # ── POLICY GRADIENT UPDATE ───────────────────────────────
        # clip BEFORE computing loss
        clipped_advantage = float(np.clip(advantage, -1.0, 1.5))

        labels = sequences[:, 1:].clone()
        labels[labels == model.t5.config.pad_token_id] = -100

        with torch.no_grad():
            ref_out = ref_model.t5(
                input_ids=input_ids.cpu(),
                attention_mask=attention_mask.cpu(),
                labels=sequences[:, 1:].clone().cpu(),
            )

        outputs = model.t5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        policy_loss = clipped_advantage * outputs.loss
        kl = max(0.0, outputs.loss.item() - ref_out.loss.item())
        loss = policy_loss + args.kl_coef * kl

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        if episode % 10 == 0:
            print(
                f"Ep {episode:5d} | reward={reward:+.4f} | "
                f"adv={advantage:+.4f} | loss={loss.item():.4f} | kl={kl:.4f}",
                flush=True,
            )

        # ── EVAL ─────────────────────────────────────────────────
        if episode % (args.eval_interval * args.batch_episodes) == 0:
            print(f"  → Evaluating...", flush=True)
            eval_scores = evaluate_rl(
                model, dev_loader, tokenizer, device,
                db_dir=DB_DIR, n_batches=150,
            )
            print(
                f"  → Eval | exec_acc={eval_scores['exec_acc']*100:.2f}% | "
                f"f1={eval_scores['f1']*100:.2f}%",
                flush=True,
            )
            if eval_scores["exec_acc"] > best_exec_acc:
                best_exec_acc = eval_scores["exec_acc"]
                model.save_checkpoint(
                    os.path.join(CHECKPOINT_DIR, "rl_best.pt"),
                    extra={"episode": episode, "exec_acc": best_exec_acc},
                )
                print(f"  ✓ New best (exec_acc={best_exec_acc*100:.2f}%)", flush=True)

            log.append({
                "episode":  episode,
                "reward":   reward,
                "exec_acc": eval_scores["exec_acc"],
                "f1":       eval_scores["f1"],
                "kl":       kl,
            })
            with open("rl_training_log.json", "w") as f:
                json.dump(log, f, indent=2)

    print(f"\nBest exec_acc: {best_exec_acc*100:.2f}%")
    print(f"Checkpoint: {CHECKPOINT_DIR}/rl_best.pt")

    if WANDB_AVAILABLE and args.use_wandb:
        wandb.finish()


# ─────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",       type=int,   default=3000)
    parser.add_argument("--batch_episodes", type=int,   default=32)
    parser.add_argument("--ppo_epochs",     type=int,   default=4)
    parser.add_argument("--lr",             type=float, default=3e-5)
    parser.add_argument("--clip_eps",       type=float, default=0.2)
    parser.add_argument("--vf_coef",        type=float, default=0.5)
    parser.add_argument("--ent_coef",       type=float, default=0.01)
    parser.add_argument("--kl_coef",        type=float, default=0.1)
    parser.add_argument("--temperature",    type=float, default=0.8)
    parser.add_argument("--eval_interval",  type=int,   default=5)
    parser.add_argument("--alpha",          type=float, default=1.0)
    parser.add_argument("--beta",           type=float, default=0.5)
    parser.add_argument("--gamma",          type=float, default=0.1)
    parser.add_argument("--delta",          type=float, default=0.3)
    parser.add_argument("--use_wandb",      action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()
    train_ppo(args)