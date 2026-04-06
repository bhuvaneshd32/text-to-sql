"""
frontend/inference_api.py
Run: python frontend/inference_api.py
"""
import os, sys, sqlite3
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer
from nlp.multi_task import TextToSQLModel
from nlp.schema_utils import load_schema_dict, serialize_schema
from config import BEST_CHECKPOINT, TABLES_JSON, DB_DIR, API_HOST, API_PORT

app = FastAPI()

# ── load both models once at startup ──────────────────────────────
print("Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained("t5-large")

print("Loading SL model...", flush=True)
sl_model = TextToSQLModel.load_for_rl(BEST_CHECKPOINT)
sl_model.eval()

RL_CHECKPOINT = os.path.join(os.path.dirname(BEST_CHECKPOINT), "rl_best.pt")
rl_model = None
if os.path.exists(RL_CHECKPOINT):
    print("Loading RL model...", flush=True)
    rl_model = TextToSQLModel.load_for_rl(RL_CHECKPOINT)
    rl_model.eval()
else:
    print(f"RL checkpoint not found at {RL_CHECKPOINT} — RL model unavailable", flush=True)

schema_dict = load_schema_dict(TABLES_JSON)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sl_model.to(device)
if rl_model:
    rl_model.to(device)

def run_sql(sql, db_path):
    try:
        conn = sqlite3.connect(db_path)
        cur  = conn.cursor()
        cur.execute(sql)
        rows    = cur.fetchall()
        columns = [d[0] for d in cur.description] if cur.description else []
        conn.close()
        return rows, columns
    except Exception as e:
        return None, []

def generate(model, question, db_id):
    schema_str = serialize_schema(db_id, schema_dict)
    prompt     = f"translate to SQL: {question} | schema: {schema_str}"
    inputs     = tokenizer(prompt, return_tensors="pt",
                           max_length=512, truncation=True).to(device)
    with torch.no_grad():
        out = model.generate_sql(
            inputs["input_ids"], inputs["attention_mask"],
            max_length=128, num_beams=4,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()

class Query(BaseModel):
    query: str
    db_id: str
    model: str = "sl"   # "sl" or "rl"

@app.post("/predict")
def predict(q: Query):
    m = rl_model if (q.model == "rl" and rl_model) else sl_model
    sql     = generate(m, q.query, q.db_id)
    db_path = os.path.join(DB_DIR, q.db_id, f"{q.db_id}.sqlite")
    rows, cols = run_sql(sql, db_path)
    return {
        "sql":          sql,
        "result_table": rows or [],
        "column_names": cols,
        "model_used":   "rl" if (q.model == "rl" and rl_model) else "sl",
        "rl_available": rl_model is not None,
    }

@app.get("/health")
def health():
    return {"status": "ok", "rl_available": rl_model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)