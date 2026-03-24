# rl/environment.py

import torch

class TextToSQLEnv:
    def __init__(
        self,
        model,              # pretrained encoder+decoder
        tokenizer,
        dataloader,
        max_sql_len=128,
        device="cpu",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.max_sql_len = max_sql_len
        self.device = device

        self.iterator = iter(dataloader)

        self.current_batch = None
        self.encoder_hidden = None
        self.partial_sql = None
        self.t = 0

    def reset(self):
        """
        Start a new episode (new example)
        """
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)

        self.current_batch = batch

        input_ids      = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        token_type_ids = batch["token_type_ids"].to(self.device)

        # Encode once (IMPORTANT)
        with torch.no_grad():
            self.encoder_hidden = self.model.encode(
                input_ids, attention_mask, token_type_ids
            )

        # Initialize SQL with BOS token
        bos_token_id = self.tokenizer.cls_token_id
        self.partial_sql = torch.tensor([[bos_token_id]], device=self.device)

        self.t = 0

        return self.get_state()

    def step(self, action):
        """
        Take one action (next token)
        """
        action = torch.tensor([[action]], device=self.device)

        # append token
        self.partial_sql = torch.cat([self.partial_sql, action], dim=1)

        self.t += 1

        done = self.is_done()

        reward = 0.0
        if done:
            reward = self.compute_reward()

        return self.get_state(), reward, done, {}

    def is_done(self):
        """
        Episode termination condition
        """
        if self.t >= self.max_sql_len:
            return True

        # EOS token check
        eos_token_id = self.tokenizer.sep_token_id
        if self.partial_sql[0, -1].item() == eos_token_id:
            return True

        return False

    def get_state(self):
        """
        Return current state representation
        """
        return {
            "encoder_hidden": self.encoder_hidden,  # [1, seq_len, 768]
            "partial_sql": self.partial_sql,        # [1, t]
        }

    def compute_reward(self):
        """
        Simple reward (placeholder)
        """
        # decode generated SQL
        pred_sql = self.tokenizer.decode(
            self.partial_sql[0].tolist(), skip_special_tokens=True
        )

        gold_sql = self.current_batch["gold_sql_ids"][0]
        gold_sql = self.tokenizer.decode(
            gold_sql.tolist(), skip_special_tokens=True
        )

        if pred_sql.strip().lower() == gold_sql.strip().lower():
            return 1.0
        return 0.0
    
# Example usage
from nlp.multi_task import TextToSQLModel
from nlp.data_pipeline import build_dataloaders
from config import TRAIN_JSON, DEV_JSON, TABLES_JSON


# ── 1. Load data ─────────────────────────────────────────────
train_loader, _, _, tokenizer = build_dataloaders(
    TRAIN_JSON,
    DEV_JSON,
    TABLES_JSON,
    batch_size=1,   # RL = 1 example per episode
)

# ── 2. Load model ────────────────────────────────────────────
model = TextToSQLModel()
model.eval()

# ── 3. Create environment ────────────────────────────────────
env = TextToSQLEnv(model, tokenizer, train_loader)

# ── 4. Run one episode ───────────────────────────────────────
state = env.reset()

done = False
steps = 0

while not done:
    action = torch.randint(0, tokenizer.vocab_size, (1,)).item()
    state, reward, done, _ = env.step(action)
    steps += 1

print(f"Episode finished in {steps} steps")
print("Reward:", reward)