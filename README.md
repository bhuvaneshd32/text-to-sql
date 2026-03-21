# Text-to-SQL — Schema-Aware Execution-Grounded RL

**Bhuvanesh D (22PD07) · ** &nbsp;|&nbsp; **Noor Fathima (22PD26) · **

A hybrid Text-to-SQL system combining a Schema-Aware Transformer Encoder with
Reinforcement Learning fine-tuning via PPO. Evaluated on the Spider benchmark.

---

## Architecture Overview

```
NL Question + DB Schema
        │
        ▼
┌───────────────────────┐
│  Schema-Aware Encoder  │  ← (nlp/)
│  RoBERTa + type embeds │
└──────────┬────────────┘
           │  Q_enc [batch, 768]
           │  S_schema [batch, n_schema, 768]
           ▼
┌───────────────────────┐
│  Grammar FSM Decoder   │  ←  (nlp/)
│  Constrained decoding  │
└──────────┬────────────┘
           │  Pretrained .pt checkpoint
           ▼
┌───────────────────────┐
│  PPO RL Fine-Tuning    │  ←  (rl/)
│  MDP + reward shaping  │
└──────────┬────────────┘
           │  /predict API
           ▼
┌───────────────────────┐
│  Streamlit Frontend    │  ←  (frontend/)
│  Query UI + heatmap    │
└───────────────────────┘
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/bhuvanesd32/text-to-sql-rl.git
cd text-to-sql-rl
pip install -r requirements.txt
```

### 2. Download Spider dataset

```bash
# Download from https://yale-lily.github.io/spider
# Extract into data/spider/ so the structure looks like:
#   data/spider/train_spider.json
#   data/spider/dev.json
#   data/spider/tables.json
#   data/spider/database/
```

### 3. Configure paths

Edit `config.py` — set `SPIDER_ROOT` if Spider is not at `data/spider/`.

---

## Running

### Verify encoder (Task 1)
```bash
python nlp/encoder.py
```

### Train (Task 6)
```bash
python nlp/train.py
```

### Evaluate (Task 7)
```bash
python nlp/eval_utils.py --checkpoint checkpoints/pretrained_best.pt
```

### Run frontend (Task 9)
```bash
streamlit run frontend/app.py
```

---

## Integration Contracts (NLP → RL)

| Module | Provider | Consumer | Format |
|---|---|---|---|
| Pretrained checkpoint | Bhuvanesh | Noor | `checkpoints/pretrained_best.pt` — keys: `model_state_dict`, `config`, `dev_ex`, `dev_f1` |
| Grammar Mask API | Bhuvanesh | Noor | `from nlp.grammar_fsm import get_mask` — `get_mask(partial_sql, db_id, tokenizer) -> BoolTensor[vocab_size]` |
| Schema linking scores | Bhuvanesh | Noor | flat vector `[n_schema_elements]` — max attention score per schema element |
| Eval utilities | Bhuvanesh | Noor | `from nlp.eval_utils import exec_accuracy, result_set_f1` |
| Inference API | Noor | Bhuvanesh | `POST /predict {query, db_id}` → `{sql, result_table, column_names, alignment_map}` |

---

## Project Structure

```
text-to-sql-rl/
├── config.py               ← ALL paths and hyperparameters here
├── requirements.txt
├── README.md
├── .gitignore
│
├── nlp/                    ← Bhuvanesh's code
│   ├── encoder.py          Task 1 — Schema-Aware Transformer Encoder
│   ├── schema_utils.py     Task 1 — serialize_schema(), load_schema_dict()
│   ├── data_pipeline.py    Task 2 — SpiderDataset, DataLoader
│   ├── cross_attention.py  Task 3 — Cross-attention schema linker
│   ├── multi_task.py       Task 4 — Dual loss (SQL + schema classification)
│   ├── grammar_fsm.py      Task 5 — SQL Grammar FSM + get_mask()
│   ├── train.py            Task 6 — Supervised pretraining loop
│   └── eval_utils.py       Task 7 — exec_accuracy(), result_set_f1()
│
├── rl/                     ← Noor's code
│   ├── environment.py      MDP environment
│   ├── reward.py           R_exec + R_sem + R_eff
│   ├── ppo.py              PPO trainer
│   ├── curriculum.py       SL→RL schedule
│   └── inference_api.py    FastAPI /predict endpoint
│
├── frontend/               ← Shared
│   ├── app.py              Streamlit app
│   └── components/
│
├── checkpoints/            ← .pt files (gitignored)
├── data/                   ← Spider dataset (gitignored)
└── scripts/
```

---

## Checkpoints

Checkpoints are **not committed to Git** (too large). Share via Google Drive or a shared folder.

Checkpoint format (Bhuvanesh → Noor):
```python
torch.save({
    "model_state_dict": ...,     
    "schema_cls_head_state_dict": ...,  
    "config": model_config_dict,
    "dev_ex": 0.52,
    "dev_f1": 0.61,
}, "checkpoints/pretrained_best.pt")
```

---

## Results

| Configuration | EX% | F1% | Exec Error Rate |
|---|---|---|---|
| SL only (Bhuvanesh baseline) | — | — | — |
| SL + grammar FSM | — | — | — |
| RL fine-tuned  | — | — | — |

*(Fill in after experiments)*