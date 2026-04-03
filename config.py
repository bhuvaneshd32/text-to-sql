"""
config.py
---------
All project-wide paths and hyperparameters in one place.
"""

import os

# ─── Data paths ──────────────────────────────────────────────────
# Point these to where you downloaded Spider
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
SPIDER_ROOT = os.environ.get("SPIDER_ROOT", os.path.join(BASE_DIR, "data", "spider"))
TABLES_JSON = os.path.join(SPIDER_ROOT, "tables.json")
TRAIN_JSON  = os.path.join(SPIDER_ROOT, "train_spider.json")
DEV_JSON    = os.path.join(SPIDER_ROOT, "dev.json")
DB_DIR      = os.path.join(SPIDER_ROOT, "database")

# ─── Model ───────────────────────────────────────────────────────
T5_MODEL      = "t5-base"
HIDDEN_SIZE        = 768
MAX_SEQ_LEN        = 512

# ─── Training ────────────────────────────────────────────────────
BATCH_SIZE         = 16
LEARNING_RATE      = 2e-5
WARMUP_STEPS       = 1000
MTL_LAMBDA         = 0.1        # weight for schema classification loss
GRAD_CLIP_NORM     = 1.0
RANDOM_SEED        = 42

# ─── Checkpoints ─────────────────────────────────────────────────
CHECKPOINT_DIR     = "checkpoints"
BEST_CHECKPOINT    = os.path.join(CHECKPOINT_DIR, "pretrained_best.pt")
LAST_CHECKPOINT    = os.path.join(CHECKPOINT_DIR, "pretrained_last.pt")

# ─── API (for frontend integration ) ────────────────────
API_HOST      = "localhost"
API_PORT      = 8000          
API_URL       = f"http://{API_HOST}:{API_PORT}"