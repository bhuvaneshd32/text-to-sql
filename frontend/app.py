"""
frontend/app.py
---------------
Streamlit frontend for Text-to-SQL demo.
Task 9: Query input + schema browser
Task 10: Attention map visualization

Run: streamlit run frontend/app.py
"""

import sys
import os
import json
import requests
import numpy as np
import torch
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TABLES_JSON,API_URL
from nlp.schema_utils import load_schema_dict, serialize_schema
from transformers import AutoTokenizer
from nlp.encoder import SchemaAwareEncoder
from nlp.cross_attention import CrossAttentionSchemaLinker, build_alignment_map, build_schema_names

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Text-to-SQL Demo",
    page_icon="🗄️",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────
# LOAD RESOURCES (cached)
# ─────────────────────────────────────────────────────────────────

@st.cache_resource
def load_resources():
    schema_dict = load_schema_dict(TABLES_JSON)
    tokenizer   = AutoTokenizer.from_pretrained("roberta-base")
    encoder     = SchemaAwareEncoder()
    linker      = CrossAttentionSchemaLinker()
    encoder.eval()
    linker.eval()
    return schema_dict, tokenizer, encoder, linker

schema_dict, tokenizer, encoder, linker = load_resources()

# ─────────────────────────────────────────────────────────────────
# SIDEBAR — schema browser
# ─────────────────────────────────────────────────────────────────

st.sidebar.title("🗄️ Schema Browser")
db_ids  = sorted(schema_dict.keys())
db_id   = st.sidebar.selectbox("Select database", db_ids, index=db_ids.index("perpetrator") if "perpetrator" in db_ids else 0)

entry   = schema_dict[db_id]
tables  = entry["tables"]
columns = entry["columns"]

st.sidebar.markdown(f"**{db_id}** — {len(tables)} tables")
for table in tables:
    cols_for_table = [c[0] for c in columns if len(c) > 2 and entry["tables"][c[2]] == table]
    with st.sidebar.expander(f"📋 {table}"):
        for col in cols_for_table:
            st.sidebar.markdown(f"  • `{col}`")

# ─────────────────────────────────────────────────────────────────
# MAIN — query input
# ─────────────────────────────────────────────────────────────────

st.title("Text-to-SQL Demo")
st.markdown(f"Database: **{db_id}**  |  Schema: `{serialize_schema(db_id, schema_dict)[:80]}...`")

question = st.text_area(
    "Ask a question about the database:",
    placeholder="e.g. How many perpetrators were killed?",
    height=80,
)

col1, col2 = st.columns([1, 4])
with col1:
    submit = st.button("Generate SQL", type="primary")
with col2:
    show_heatmap = st.checkbox("Show attention heatmap")

# ─────────────────────────────────────────────────────────────────
# ON SUBMIT — call Noor's API or local model
# ─────────────────────────────────────────────────────────────────

if submit and question.strip():
    st.markdown("---")

    # try Noor's /predict endpoint
    try:
        resp = requests.post(
            f"{NOOR_API_URL}/predict",
            json={"query": question, "db_id": db_id},
            timeout=10,
        )
        data = resp.json()
        pred_sql      = data.get("sql", "")
        result_table  = data.get("result_table", [])
        column_names  = data.get("column_names", [])
        alignment_map = data.get("alignment_map", None)
        api_used      = "Noor's RL model"

    except Exception as e:
        st.warning(f"Noor's API not reachable ({e}). Showing attention map only.")
        pred_sql     = "(API unavailable — run Noor's inference_api.py)"
        result_table = []
        column_names = []
        alignment_map = None
        api_used      = "local"

    # display SQL
    st.subheader("Generated SQL")
    st.caption(f"Source: {api_used}")
    st.code(pred_sql, language="sql")

    # display result table
    if result_table and column_names:
        st.subheader("Query Results")
        df = pd.DataFrame(result_table, columns=column_names)
        st.dataframe(df, use_container_width=True)
    elif result_table:
        st.subheader("Query Results")
        st.dataframe(pd.DataFrame(result_table), use_container_width=True)

    # ── attention heatmap (Task 10) ───────────────────────────────
    if show_heatmap:
        st.subheader("Attention Map — Question → Schema")
        st.caption("Darker = stronger attention. Shows which question words relate to which schema columns.")

        with st.spinner("Computing attention..."):
            schema_str = serialize_schema(db_id, schema_dict)
            q_tokens   = tokenizer(question, add_special_tokens=False)["input_ids"]
            q_len      = len(q_tokens)

            encoding = tokenizer(
                question, schema_str,
                add_special_tokens=True,
                max_length=512, truncation=True, padding="max_length",
                return_tensors="pt",
            )
            input_ids      = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]
            seq_len        = input_ids.shape[1]

            token_type_ids = torch.zeros(1, seq_len, dtype=torch.long)
            schema_start   = q_len + 2
            for i in range(schema_start, seq_len):
                if attention_mask[0, i] == 0:
                    break
                token_type_ids[0, i] = 1

            with torch.no_grad():
                outputs    = encoder.roberta(
                    inputs_embeds=encoder.roberta.embeddings.word_embeddings(input_ids)
                                  + encoder.type_embedding(token_type_ids),
                    attention_mask=attention_mask,
                )
                all_hidden = outputs.last_hidden_state
                H_q        = all_hidden[:, 1:q_len+1, :]
                _, S_schema = encoder(input_ids, attention_mask, token_type_ids)[:2]
                _, A, _    = linker(H_q, S_schema)

            q_token_strings = tokenizer.convert_ids_to_tokens(
                input_ids[0, 1:q_len+1].tolist()
            )
            q_token_strings = [t.lstrip("Ġ").lstrip("ġ") for t in q_token_strings]
            schema_names    = build_schema_names(db_id, schema_dict)

            A_np = A[0].detach().cpu().numpy()
            n_q  = min(len(q_token_strings), A_np.shape[0])
            n_s  = min(len(schema_names), A_np.shape[1], 30)  # cap at 30 schema cols

            fig, ax = plt.subplots(figsize=(max(10, n_s * 0.5), max(4, n_q * 0.5)))
            sns.heatmap(
                A_np[:n_q, :n_s],
                xticklabels=schema_names[:n_s],
                yticklabels=q_token_strings[:n_q],
                cmap="Blues",
                ax=ax,
                vmin=0, vmax=A_np[:n_q, :n_s].max(),
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
            ax.set_xlabel("Schema elements")
            ax.set_ylabel("Question tokens")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ─────────────────────────────────────────────────────────────────
# TRAINING DASHBOARD (reads training_log.json)
# ─────────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("Training Dashboard")

log_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "nlp", "training_log.json"
)

if os.path.exists(log_path):
    with open(log_path) as f:
        log = json.load(f)
    if log:
        epochs  = [e["epoch"]      for e in log]
        losses  = [e["train_loss"] for e in log]
        dev_exs = [e["dev_ex"]     for e in log]

        c1, c2 = st.columns(2)
        with c1:
            st.line_chart(pd.DataFrame({"train loss": losses}, index=epochs))
            st.caption("Training loss per epoch")
        with c2:
            st.line_chart(pd.DataFrame({"dev exact match": dev_exs}, index=epochs))
            st.caption("Dev exact match per epoch")

        latest = log[-1]
        m1, m2, m3 = st.columns(3)
        m1.metric("Latest epoch",    latest["epoch"])
        m2.metric("Train loss",      f"{latest['train_loss']:.4f}")
        m3.metric("Dev exact match", f"{latest['dev_ex']*100:.1f}%")
else:
    st.info("training_log.json not found — run train.py first.")