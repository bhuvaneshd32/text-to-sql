# rl/environment.py

import os
import re

import torch
from rl.reward import compute_reward
from nlp.grammar_fsm import SQLGrammarFSM
from nlp.schema_utils import load_schema_dict
from config import BEST_CHECKPOINT, TABLES_JSON

def normalize_sql(sql):
    sql = re.sub(r"\s+", " ", sql)
    sql = sql.replace("(", " ( ").replace(")", " ) ")
    sql = sql.replace(",", " , ")
    return sql.lower().strip()

class TextToSQLEnv:
    def __init__(
        self,
        model,
        tokenizer,
        dataloader,
        schema_dict,
        max_sql_len=128,
        device="cpu",
    ):
        self.model       = model
        self.tokenizer   = tokenizer
        self.dataloader  = dataloader
        self.schema_dict = schema_dict
        self.max_sql_len = max_sql_len
        self.device      = device

        self.data_iter        = iter(dataloader)
        self.current_example  = None
        self.current_memory   = None
        self.generated_tokens = []
        self.t                = 0
        self.fsm              = None

        # T5 EOS token id — used for episode termination
        self._eos_id = self.tokenizer.eos_token_id

    def decode_sql(self, token_ids):
        """
        Convert token ids to SQL string.
        T5 uses SentencePiece with ▁ prefix (not RoBERTa's Ġ).
        We decode the full sequence at once for correct reconstruction.
        """
        if len(token_ids) == 0:
            return ""
        if isinstance(token_ids[0], list):
            token_ids = [t for sub in token_ids for t in sub]

        # Decode full sequence at once — T5 SentencePiece handles
        # word boundaries correctly this way
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return " ".join(text.split()).strip()

    def get_state(self):
        return {
            "encoder_hidden": self.current_memory,
            "partial_sql":    self.generated_tokens,
        }

    def reset(self):
        """Start a new episode. Caches encoder output."""
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            batch = next(self.data_iter)

        self.current_example  = batch
        self.generated_tokens = []
        self.t                = 0

        device = next(self.model.parameters()).device
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            self.current_memory = self.model.encode(
                input_ids, attention_mask
            )

        db_id = batch["db_ids"][0]
        self.fsm = SQLGrammarFSM(
            db_id=db_id,
            schema_dict=self.schema_dict,
            tokenizer=self.tokenizer,
        )

        return self.get_state()

    def step(self, action):
        """
        Take one action (generate next token).
        Terminal reward = compute_reward() at end of episode.
        Intermediate reward = small shaped signal for structure milestones.
        """
        partial_sql = normalize_sql(self.decode_sql(self.generated_tokens))

        mask       = self.fsm.get_mask(partial_sql)
        vocab_size = self.model.t5.config.vocab_size 

        # pad mask to actual tokenizer vocab size
        if mask.shape[0] < vocab_size:
            pad  = torch.zeros(vocab_size - mask.shape[0], dtype=torch.bool)
            mask = torch.cat([mask, pad])


        # Force EOS if episode running too long
        if self.t > 60:
            eos_id = self._eos_id
            if eos_id is not None and eos_id < vocab_size:
                mask[:] = False
                mask[eos_id] = True

        # Force FROM if needed
        if "from" not in partial_sql and self.t > 10:
            from_token_ids = self.tokenizer.encode(" from", add_special_tokens=False)
            for tid in from_token_ids:
                if tid < vocab_size:
                    mask[tid] = True

        # Handle out-of-bounds
        if action < 0 or action >= vocab_size:
            return self.get_state(), -1.0, True, {"error": "action_out_of_bounds"}

        # Handle grammar violation
        if not mask[action]:
            # soft termination — no penalty, just end episode
            # avoids punishing mask disagreement between collect and step
            return self.get_state(), 0.0, True, {"error": "invalid_action"}
        
        # Enforce SELECT as first token (T5-safe check)
        if self.t == 0:
            token_str = self.tokenizer.decode(
                [action], skip_special_tokens=True
            ).lower().strip()
            if "select" not in token_str:
                return self.get_state(), 0.0, True, {"error": "must_start_with_select"}

        # Apply action
        self.generated_tokens.append(int(action))
        self.t += 1

        new_partial = normalize_sql(self.decode_sql(self.generated_tokens))

        done = self.is_done()

        if done:
            pred_sql = self.decode_sql(self.generated_tokens)
            gold_sql_ids = self.current_example["gold_sql_ids"][0]
            gold_ids = [tid for tid in gold_sql_ids.tolist() if tid >= 0]
            gold_sql = self.tokenizer.decode(
                gold_ids, skip_special_tokens=True
            )
            db_id   = self.current_example["db_ids"][0]
            db_path = os.path.join(
                __import__('config').DB_DIR, db_id, f"{db_id}.sqlite"
            )

            reward = compute_reward(
                pred_sql=pred_sql,
                gold_sql=gold_sql,
                db_path=db_path,
                sql_tokens=self.generated_tokens,
            )

            # Only print every 50 episodes to reduce noise
            if self.t % 50 == 1:
                print(f"\n--- DEBUG ---")
                print(f"Pred: {pred_sql[:100]}")
                print(f"Gold: {gold_sql[:100]}")
                print(f"DB:   {db_id}")

        else:
            # Shaped intermediate reward for structural milestones
            shaped = 0.0
            prev_partial = normalize_sql(
                self.decode_sql(self.generated_tokens[:-1])
            )
            if "from" in new_partial and "from" not in prev_partial:
                has_col = any(col in prev_partial for col in self.fsm.columns)
                shaped += 0.05 if has_col else 0.01
            if "where" in new_partial and "where" not in prev_partial:
                shaped += 0.03
            reward = shaped

        return self.get_state(), reward, done, {}

    def is_done(self):
        if self.t >= self.max_sql_len:
            return True
        if len(self.generated_tokens) > 0:
            # T5 uses eos_token_id — sep_token_id is None in T5
            if self.generated_tokens[-1] == self._eos_id:
                return True
        return False


if __name__ == "__main__":
    import re
    from nlp.multi_task import TextToSQLModel
    from nlp.data_pipeline import build_dataloaders
    from config import DB_DIR, TRAIN_JSON, DEV_JSON, TABLES_JSON

    train_loader, _, _, tokenizer = build_dataloaders(
        TRAIN_JSON, DEV_JSON, TABLES_JSON, batch_size=1,
    )
    model = TextToSQLModel.load_for_rl(
        BEST_CHECKPOINT)
    model.eval()
    schema_dict = load_schema_dict(TABLES_JSON)
    env = TextToSQLEnv(model, tokenizer, train_loader, schema_dict)

    state = env.reset()
    done  = False
    steps = 0


    while not done:
        memory = env.current_memory

        # Always fetch mask fresh after last action
        partial_sql = normalize_sql(env.decode_sql(env.generated_tokens))
        mask = env.fsm.get_mask(partial_sql)

        logits = model.decode_step(memory, env.generated_tokens)
        logits = logits.float()
        logits[~mask] = -1e9

        temperature = 0.7 if env.t > 5 else 0.3
        probs = torch.softmax(logits / temperature, dim=-1)

        if probs.sum() < 1e-9 or torch.isnan(probs).any():
            action = logits.argmax().item()
        else:
            action = torch.multinomial(probs, num_samples=1).item()

        state, reward, done, _ = env.step(action)
        steps += 1

    print(f"Episode finished in {steps} steps, Reward: {reward}")