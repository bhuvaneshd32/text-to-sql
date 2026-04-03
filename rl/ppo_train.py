"""
rl/ppo_train.py
---------------
PPO fine-tuning loop for Text-to-SQL.

Builds on top of:
    - TextToSQLModel (multi_task.py)  — policy network
    - TextToSQLEnv   (environment.py) — MDP wrapper
    - compute_reward (reward.py)      — R_exec + R_sem + R_eff
    - exec_accuracy, result_set_f1 from nlp.eval_utils

PPO components:
    - Clipped surrogate objective
    - GAE advantage estimation
    - Entropy bonus (exploration)
    - KL penalty against SL checkpoint (prevent forgetting)
    - Value head (small MLP on decoder hidden state)

RUN:
    python -m rl.ppo_train --episodes 50 --batch_episodes 8   # sanity check
    python -m rl.ppo_train --episodes 3000 --use_wandb        # full training
"""

import os
import sys
import json
import re
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.distributions import Categorical

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    TRAIN_JSON, DEV_JSON, TABLES_JSON, DB_DIR,
    CHECKPOINT_DIR, LAST_CHECKPOINT,
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

BEST_CHECKPOINT = LAST_CHECKPOINT  # start PPO from last SL checkpoint
# ─────────────────────────────────────────────────────────────────
# VALUE HEAD
# ─────────────────────────────────────────────────────────────────

class ValueHead(nn.Module):
    """
    Small MLP that estimates V(s) from decoder's last hidden state.

    Input:  [batch, 768]  last hidden state from decoder
    Output: [batch]       scalar value estimate
    """
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden).squeeze(-1)


# ─────────────────────────────────────────────────────────────────
# DECODE STEP — returns hidden state alongside logits
# ─────────────────────────────────────────────────────────────────

def decode_step_with_hidden(model, memory, generated_tokens, device):
    """
    Same as model.decode_step() but also returns last_hidden [1, 768].
    Value head needs last_hidden to estimate V(s).
    PPO loss reuses last_hidden to recompute log probs without
    re-running the full decoder.

    Returns:
        logits:      [vocab_size]
        last_hidden: [1, 768]
    """
    if len(generated_tokens) == 0:
        input_ids = torch.tensor(
            [[model.tokenizer.cls_token_id]], device=device
        )
    else:
        input_ids = torch.tensor([generated_tokens], device=device)

    seq_len    = input_ids.shape[1]
    positions  = torch.arange(seq_len, device=device).unsqueeze(0)
    dec_embeds = (
        model.token_embedding(input_ids)
        + model.pos_embedding(positions)
    )
    causal_mask  = nn.Transformer.generate_square_subsequent_mask(
        seq_len, device=device
    )
    enc_pad_mask = torch.zeros(
        memory.shape[:2], dtype=torch.bool, device=device
    )
    dec_hidden  = model.decoder(
        tgt=dec_embeds,
        memory=memory,
        tgt_mask=causal_mask,
        memory_key_padding_mask=enc_pad_mask,
    )
    last_hidden = dec_hidden[:, -1, :]                             # [1, 768]
    logits      = model.output_projection(last_hidden).squeeze(0)  # [vocab]
    return logits, last_hidden


# ─────────────────────────────────────────────────────────────────
# GAE — Generalised Advantage Estimation
# ─────────────────────────────────────────────────────────────────

def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """
    Compute GAE advantages and discounted returns.

    Args:
        rewards: list[float]  per-step rewards (0 until final step)
        values:  list[float]  V(s) estimates from value head
        gamma:   discount factor
        lam:     GAE lambda — bias/variance tradeoff

    Returns:
        advantages: Tensor [T]
        returns:    Tensor [T]   target for value loss
    """
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

    if advantages.std() > 1e-8:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns


# ─────────────────────────────────────────────────────────────────
# COLLECT TRAJECTORY — one full episode
# ─────────────────────────────────────────────────────────────────

def normalize_sql(sql):
    sql = re.sub(r"\s+", " ", sql)
    sql = sql.replace("(", " ( ").replace(")", " ) ")
    sql = sql.replace(",", " , ")
    return sql.lower().strip()


def collect_trajectory(env, model, value_head, device, temperature=0.8):
    """
    Run one full episode under the current policy.
    Reward is 0 at every intermediate step.
    Terminal reward = R_exec + R_sem + R_eff.
    GAE propagates terminal reward back through the episode.
    """
    state  = env.reset()
    memory = env.current_memory.to(device)

    actions, log_probs, values, rewards, hiddens = [], [], [], [], []
    done = False

    while not done:
        with torch.no_grad():
            logits, last_hidden = decode_step_with_hidden(
                model, memory, env.generated_tokens, device
            )
            value = value_head(last_hidden).item()

        # apply FSM grammar mask
        partial = normalize_sql(env.decode_sql(env.generated_tokens))
        mask    = env.fsm.get_mask(partial)

        logits        = logits.float()
        logits[~mask] = -1e9

        probs = F.softmax(logits / temperature, dim=-1)

        # safety: collapsed or NaN probs → argmax fallback
        if probs.sum() < 1e-9 or torch.isnan(probs).any():
            action = logits.argmax().item()
            lp     = 0.0
        else:
            dist   = Categorical(probs)
            action = dist.sample().item()
            lp     = dist.log_prob(torch.tensor(action)).item()

        _, reward, done, _ = env.step(action)

        actions.append(action)
        log_probs.append(lp)
        values.append(value)
        rewards.append(reward)
        hiddens.append(last_hidden.squeeze(0).detach().cpu())

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
    model,
    value_head,
    ref_model,
    optimizer,
    trajectories,
    device,
    clip_eps   = 0.2,
    vf_coef    = 0.5,
    ent_coef   = 0.01,
    kl_coef    = 0.1,
    ppo_epochs = 4,
):
    """
    PPO gradient update over a batch of trajectories.

    Loss = L_policy + vf_coef * L_value - ent_coef * entropy + kl_coef * KL

    L_policy:  clipped surrogate — prevents large destructive updates
    L_value:   MSE between predicted V(s) and GAE returns
    entropy:   encourages exploration
    KL:        prevents forgetting supervised pretraining
    """
    # ── flatten all trajectories ──────────────────────────────────
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
    hiddens    = torch.stack(all_hiddens).to(device)   # [T, 768]

    loss_log = []

    for _ in range(ppo_epochs):

        # recompute logits from stored hidden states
        new_logits    = model.output_projection(hiddens)       # [T, vocab]
        new_log_probs = F.log_softmax(new_logits, dim=-1)
        new_lps       = new_log_probs.gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)                                           # [T]

        # clipped policy loss
        ratio    = torch.exp(new_lps - old_lps)
        surr1    = ratio * advantages
        surr2    = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        L_policy = -torch.min(surr1, surr2).mean()

        # value loss
        pred_values = value_head(hiddens).squeeze(-1)          # [T]
        L_value     = F.mse_loss(pred_values, returns)

        # entropy bonus — prevents policy collapse
        probs   = F.softmax(new_logits, dim=-1)
        entropy = -(probs * (new_log_probs + 1e-8)).sum(dim=-1).mean()

        # KL penalty vs frozen SL reference
        # prevents drifting too far from supervised pretraining
        with torch.no_grad():
            ref_logits = ref_model.output_projection(hiddens)
        ref_probs = F.softmax(ref_logits, dim=-1)
        kl_div    = F.kl_div(
            F.log_softmax(new_logits, dim=-1),
            ref_probs,
            reduction="batchmean",
        )

        loss = (
            L_policy
            + vf_coef  * L_value
            - ent_coef * entropy
            + kl_coef  * kl_div
        )

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
# EVALUATION — uses Bhuvanesh's shared metrics
# ─────────────────────────────────────────────────────────────────

def evaluate_rl(model, dev_loader, tokenizer, device, db_dir, n_batches=30):
    """
    Quick evaluation during RL training.
    Uses exec_accuracy and result_set_f1 from nlp.eval_utils so
    SL vs RL comparison is apples-to-apples.
    """
    model.eval()
    ex_scores, f1_scores = [], []

    with torch.no_grad():
        for i, batch in enumerate(dev_loader):
            if i >= n_batches:
                break

            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            db_id          = batch["db_ids"][0]
            gold_sql_ids   = batch["gold_sql_ids"][0]

            memory = model.encode(input_ids, attention_mask, token_type_ids)

            # greedy decode — no sampling during eval
            generated = []
            for _ in range(128):
                logits, _ = decode_step_with_hidden(
                    model, memory, generated, device
                )
                token = logits.argmax().item()
                if token == tokenizer.sep_token_id:
                    break
                generated.append(token)

            pred_sql = tokenizer.decode(
                generated, skip_special_tokens=True
            ).strip()
            gold_sql = tokenizer.decode(
                gold_sql_ids.tolist(), skip_special_tokens=True
            ).strip()

            db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
            ex_scores.append(exec_accuracy(pred_sql, gold_sql, db_path))
            f1_scores.append(result_set_f1(pred_sql, gold_sql, db_path))

    model.train()
    return {
        "exec_acc": float(np.mean(ex_scores)) if ex_scores else 0.0,
        "f1":       float(np.mean(f1_scores)) if f1_scores else 0.0,
    }


# ─────────────────────────────────────────────────────────────────
# CURRICULUM TIER — for logging
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
# MAIN PPO TRAINING LOOP
# ─────────────────────────────────────────────────────────────────

def train_ppo(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"PPO Fine-tuning — Text-to-SQL")
    print(f"{'='*60}")
    print(f"Device:          {device}")
    print(f"Episodes:        {args.episodes}")
    print(f"Batch episodes:  {args.batch_episodes}")
    print(f"PPO epochs:      {args.ppo_epochs}")
    print(f"LR:              {args.lr}")
    print(f"clip_eps:        {args.clip_eps}")
    print(f"ent_coef:        {args.ent_coef}")
    print(f"kl_coef:         {args.kl_coef}")
    print(f"{'='*60}\n")

    # ── data ──────────────────────────────────────────────────────
    train_loader, dev_loader, _, tokenizer = build_dataloaders(
        TRAIN_JSON, DEV_JSON, TABLES_JSON, batch_size=1
    )
    schema_dict = load_schema_dict(TABLES_JSON)

    # ── policy model ──────────────────────────────────────────────
    model = TextToSQLModel.load_for_rl(
        BEST_CHECKPOINT, tokenizer=tokenizer
    )
    model.to(device)
    model.train()

    # freeze encoder — protect Bhuvanesh's trained representations
    # only decoder, output projection, token embeddings updated by PPO
    for p in model.encoder.parameters():
        p.requires_grad = False
    print("Encoder frozen — only decoder updated by PPO.\n")

    # ── frozen reference model for KL penalty ────────────────────
    ref_model = TextToSQLModel.load_for_rl(
        BEST_CHECKPOINT, tokenizer=tokenizer
    )
    ref_model.to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # ── value head ────────────────────────────────────────────────
    value_head = ValueHead(hidden_size=768).to(device)

    # ── optimizer ─────────────────────────────────────────────────
    # Lower LR than supervised — PPO updates must be small
    optimizer = AdamW(
        list(model.decoder.parameters())
        + list(model.output_projection.parameters())
        + list(model.token_embedding.parameters())
        + list(value_head.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )

    # ── environment ───────────────────────────────────────────────
    env = TextToSQLEnv(model, tokenizer, train_loader, schema_dict)

    # ── setup ─────────────────────────────────────────────────────
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    log          = []
    best_exec_acc= 0.0
    batch_trajs  = []

    if WANDB_AVAILABLE and args.use_wandb:
        wandb.init(project="text2sql-ppo", config=vars(args))

    print("Starting PPO training...\n")

    for episode in range(1, args.episodes + 1):

        # ── collect one episode ───────────────────────────────────
        traj = collect_trajectory(
            env, model, value_head, device,
            temperature=args.temperature,
        )
        batch_trajs.append(traj)

        # ── episode logging every 10 ──────────────────────────────
        if episode % 10 == 0:
            recent     = batch_trajs[-min(10, len(batch_trajs)):]
            avg_reward = np.mean([t["final_reward"] for t in recent])
            avg_steps  = np.mean([t["steps"]        for t in recent])
            tier       = get_curriculum_tier(episode)
            print(
                f"Ep {episode:5d} | "
                f"reward={avg_reward:+.4f} | "
                f"steps={avg_steps:.1f} | "
                f"tier={tier}",
                flush=True,
            )

        # ── PPO update every batch_episodes ───────────────────────
        if len(batch_trajs) >= args.batch_episodes:

            loss_info  = ppo_update(
                model, value_head, ref_model,
                optimizer, batch_trajs, device,
                clip_eps   = args.clip_eps,
                vf_coef    = args.vf_coef,
                ent_coef   = args.ent_coef,
                kl_coef    = args.kl_coef,
                ppo_epochs = args.ppo_epochs,
            )
            avg_reward = np.mean([t["final_reward"] for t in batch_trajs])

            print(
                f"  → PPO update | "
                f"loss={loss_info['loss']:.4f} | "
                f"policy={loss_info['policy']:.4f} | "
                f"value={loss_info['value']:.4f} | "
                f"entropy={loss_info['entropy']:.4f} | "
                f"kl={loss_info['kl']:.4f} | "
                f"avg_reward={avg_reward:.4f}",
                flush=True,
            )

            batch_trajs = []

            # ── evaluate every eval_interval updates ──────────────
            update_num = episode // args.batch_episodes
            if update_num % args.eval_interval == 0:
                print(f"  → Evaluating...", flush=True)
                eval_scores = evaluate_rl(
                    model, dev_loader, tokenizer,
                    device, db_dir=DB_DIR, n_batches=30,
                )
                print(
                    f"  → Eval | "
                    f"exec_acc={eval_scores['exec_acc']*100:.2f}% | "
                    f"f1={eval_scores['f1']*100:.2f}%",
                    flush=True,
                )

                # save best checkpoint
                if eval_scores["exec_acc"] > best_exec_acc:
                    best_exec_acc = eval_scores["exec_acc"]
                    model.save_checkpoint(
                        os.path.join(CHECKPOINT_DIR, "rl_best.pt"),
                        extra={
                            "episode":  episode,
                            "exec_acc": best_exec_acc,
                            "f1":       eval_scores["f1"],
                        },
                    )
                    print(
                        f"  ✓ New best RL checkpoint  "
                        f"(exec_acc={best_exec_acc*100:.2f}%)",
                        flush=True,
                    )

                # log
                entry = {
                    "episode":    episode,
                    "avg_reward": avg_reward,
                    "exec_acc":   eval_scores["exec_acc"],
                    "f1":         eval_scores["f1"],
                    **loss_info,
                }
                log.append(entry)
                with open("rl_training_log.json", "w") as f:
                    json.dump(log, f, indent=2)

                if WANDB_AVAILABLE and args.use_wandb:
                    wandb.log({
                        "episode":    episode,
                        "avg_reward": avg_reward,
                        "exec_acc":   eval_scores["exec_acc"],
                        "f1":         eval_scores["f1"],
                        **loss_info,
                    })

                # KL warning
                if loss_info["kl"] > 0.5:
                    print(
                        f"  [WARNING] KL={loss_info['kl']:.4f} > 0.5 — "
                        f"policy drifting. Increase --kl_coef.",
                        flush=True,
                    )

    # ── done ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PPO training complete.")
    print(f"Best exec_acc: {best_exec_acc*100:.2f}%")
    print(f"Checkpoint:    {CHECKPOINT_DIR}/rl_best.pt")

    if WANDB_AVAILABLE and args.use_wandb:
        wandb.finish()


# ─────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",       type=int,   default=3000)
    parser.add_argument("--batch_episodes", type=int,   default=16,
                        help="Episodes per PPO update")
    parser.add_argument("--ppo_epochs",     type=int,   default=4,
                        help="Gradient steps per batch")
    parser.add_argument("--lr",             type=float, default=1e-5,
                        help="Lower than SL — PPO updates must be small")
    parser.add_argument("--clip_eps",       type=float, default=0.2)
    parser.add_argument("--vf_coef",        type=float, default=0.5)
    parser.add_argument("--ent_coef",       type=float, default=0.01)
    parser.add_argument("--kl_coef",        type=float, default=0.1)
    parser.add_argument("--temperature",    type=float, default=0.8)
    parser.add_argument("--eval_interval",  type=int,   default=5,
                        help="Evaluate every N PPO updates")
    parser.add_argument("--alpha", type=float, default=1.0,
                    help="Weight for R_exec")
    parser.add_argument("--beta",  type=float, default=0.5, help="Weight for R_sem")
    parser.add_argument("--gamma", type=float, default=0.1,help="Weight for R_eff")
    parser.add_argument("--delta", type=float, default=0.3, help="Weight for R_counterfactual")
    parser.add_argument("--use_wandb",      action="store_true")
    args = parser.parse_args()
    train_ppo(args)