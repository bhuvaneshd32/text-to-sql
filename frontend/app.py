"""
frontend/app.py
---------------
Streamlit frontend for Text-to-SQL demo.
Includes SL vs RL model comparison and RL training dashboard.

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
from config import TABLES_JSON, API_URL
from nlp.schema_utils import load_schema_dict, serialize_schema

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
    return schema_dict

schema_dict = load_resources()

# ─────────────────────────────────────────────────────────────────
# CHECK API + RL AVAILABILITY
# ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        data = r.json()
        return True, data.get("rl_available", False)
    except Exception:
        return False, False

api_online, rl_available = check_api()

# ─────────────────────────────────────────────────────────────────
# SIDEBAR — schema browser + model selector
# ─────────────────────────────────────────────────────────────────

st.sidebar.title("🗄️ Schema Browser")
db_ids = sorted(schema_dict.keys())
db_id  = st.sidebar.selectbox(
    "Select database", db_ids,
    index=db_ids.index("perpetrator") if "perpetrator" in db_ids else 0
)

entry   = schema_dict[db_id]
tables  = entry["tables"]
columns = entry["columns"]

st.sidebar.markdown(f"**{db_id}** — {len(tables)} tables")
for table in tables:
    cols_for_table = [c[0] for c in columns if len(c) > 2 and entry["tables"][c[2]] == table]
    with st.sidebar.expander(f"📋 {table}"):
        for col in cols_for_table:
            st.sidebar.markdown(f"  • `{col}`")

st.sidebar.markdown("---")
st.sidebar.title("🤖 Model")

# API status indicator
if api_online:
    st.sidebar.success("API online ✓")
else:
    st.sidebar.error("API offline — start inference_api.py")

# Model selector
model_options = ["SL (pretrained T5-large)"]
if rl_available:
    model_options.append("RL fine-tuned")
    model_options.append("Compare SL vs RL")

selected_model = st.sidebar.radio("Model to use", model_options)

if not rl_available and api_online:
    st.sidebar.info("RL checkpoint not found.\nRun RL training first:\n`python -m rl.ppo_train`")

# ─────────────────────────────────────────────────────────────────
# MAIN — query input
# ─────────────────────────────────────────────────────────────────

st.title("Text-to-SQL Demo")
st.markdown(
    f"Database: **{db_id}**  |  "
    f"Schema: `{serialize_schema(db_id, schema_dict)[:80]}...`"
)

question = st.text_area(
    "Ask a question about the database:",
    placeholder="e.g. How many perpetrators were killed?",
    height=80,
)

col1, col2 = st.columns([1, 4])
with col1:
    submit = st.button("Generate SQL", type="primary", disabled=not api_online)
with col2:
    show_heatmap = st.checkbox("Show attention heatmap", disabled=True,
                               help="Attention heatmap requires RoBERTa encoder — disabled for T5")

# ─────────────────────────────────────────────────────────────────
# HELPER — call API
# ─────────────────────────────────────────────────────────────────

def call_api(question, db_id, model="sl"):
    resp = requests.post(
        f"{API_URL}/predict",
        json={"query": question, "db_id": db_id, "model": model},
        timeout=30,
    )
    return resp.json()

# ─────────────────────────────────────────────────────────────────
# ON SUBMIT
# ─────────────────────────────────────────────────────────────────

if submit and question.strip():
    st.markdown("---")

    if "Compare" in selected_model:
        # ── SIDE BY SIDE SL vs RL ─────────────────────────────────
        st.subheader("SL vs RL Comparison")

        col_sl, col_rl = st.columns(2)

        with st.spinner("Generating from both models..."):
            try:
                sl_data = call_api(question, db_id, model="sl")
                rl_data = call_api(question, db_id, model="rl")
            except Exception as e:
                st.error(f"API error: {e}")
                st.stop()

        with col_sl:
            st.markdown("#### 🔵 SL Model (pretrained)")
            st.code(sl_data.get("sql", ""), language="sql")
            rows = sl_data.get("result_table", [])
            cols = sl_data.get("column_names", [])
            if rows:
                st.dataframe(
                    pd.DataFrame(rows, columns=cols) if cols else pd.DataFrame(rows),
                    use_container_width=True
                )
            else:
                st.info("No results returned.")

        with col_rl:
            st.markdown("#### 🟢 RL Model (fine-tuned)")
            st.code(rl_data.get("sql", ""), language="sql")
            rows = rl_data.get("result_table", [])
            cols = rl_data.get("column_names", [])
            if rows:
                st.dataframe(
                    pd.DataFrame(rows, columns=cols) if cols else pd.DataFrame(rows),
                    use_container_width=True
                )
            else:
                st.info("No results returned.")

        # highlight if they differ
        if sl_data.get("sql", "").strip().lower() != rl_data.get("sql", "").strip().lower():
            st.warning("⚡ SL and RL models generated different SQL — results may differ.")
        else:
            st.success("✓ Both models generated identical SQL.")

    else:
        # ── SINGLE MODEL ─────────────────────────────────────────
        model_key = "rl" if "RL" in selected_model else "sl"
        model_label = "🟢 RL fine-tuned" if model_key == "rl" else "🔵 SL pretrained"

        try:
            with st.spinner(f"Generating SQL with {model_label}..."):
                data = call_api(question, db_id, model=model_key)
        except Exception as e:
            st.error(f"API error: {e}")
            st.stop()

        pred_sql     = data.get("sql", "")
        result_table = data.get("result_table", [])
        column_names = data.get("column_names", [])

        st.subheader("Generated SQL")
        st.caption(f"Model: {model_label}")
        st.code(pred_sql, language="sql")

        if result_table:
            st.subheader("Query Results")
            df = pd.DataFrame(result_table, columns=column_names) if column_names \
                 else pd.DataFrame(result_table)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Query returned no results or failed to execute.")

# ─────────────────────────────────────────────────────────────────
# TRAINING DASHBOARD
# ─────────────────────────────────────────────────────────────────

st.markdown("---")

tab_sl, tab_rl = st.tabs(["📈 SL Training", "🎯 RL Training"])

# ── SL training log ───────────────────────────────────────────────
with tab_sl:
    sl_log_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "nlp", "training_log.json"
    )
    if os.path.exists(sl_log_path):
        with open(sl_log_path) as f:
            sl_log = json.load(f)
        if sl_log:
            epochs  = [e["epoch"]      for e in sl_log]
            losses  = [e["train_loss"] for e in sl_log]
            dev_exs = [e["dev_ex"]     for e in sl_log]

            c1, c2 = st.columns(2)
            with c1:
                st.line_chart(pd.DataFrame({"train loss": losses}, index=epochs))
                st.caption("Training loss per epoch")
            with c2:
                st.line_chart(pd.DataFrame({"dev exact match": dev_exs}, index=epochs))
                st.caption("Dev exact match per epoch")

            latest = sl_log[-1]
            m1, m2, m3 = st.columns(3)
            m1.metric("Latest epoch",    latest["epoch"])
            m2.metric("Train loss",      f"{latest['train_loss']:.4f}")
            m3.metric("Dev exact match", f"{latest['dev_ex']*100:.1f}%")
    else:
        st.info("training_log.json not found — run nlp/train.py first.")

# ── RL training log ───────────────────────────────────────────────
with tab_rl:
    rl_log_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "rl_training_log.json"
    )
    if os.path.exists(rl_log_path):
        with open(rl_log_path) as f:
            rl_log = json.load(f)
        if rl_log:
            episodes  = [e["episode"]  for e in rl_log]
            rewards   = [e["reward"]   for e in rl_log]
            exec_accs = [e["exec_acc"] * 100 for e in rl_log]
            f1s       = [e["f1"]       * 100 for e in rl_log]
            kls       = [e["kl"]       for e in rl_log]

            # metrics over episodes
            c1, c2 = st.columns(2)
            with c1:
                st.line_chart(pd.DataFrame({"exec_acc %": exec_accs, "f1 %": f1s},
                                           index=episodes))
                st.caption("Exec accuracy and F1 over RL episodes")
            with c2:
                st.line_chart(pd.DataFrame({"reward": rewards, "kl": kls},
                                           index=episodes))
                st.caption("Reward and KL divergence over RL episodes")

            latest_rl = rl_log[-1]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Episodes trained", latest_rl["episode"])
            m2.metric("Exec acc",  f"{latest_rl['exec_acc']*100:.2f}%")
            m3.metric("F1",        f"{latest_rl['f1']*100:.2f}%")
            m4.metric("KL",        f"{latest_rl['kl']:.4f}")

            # SL vs RL comparison metrics
            st.markdown("#### SL vs RL Baseline Comparison")
            comp_data = {
                "Model":    ["SL pretrained", "RL fine-tuned"],
                "Exec Acc": ["33.75%", f"{latest_rl['exec_acc']*100:.2f}%"],
                "F1":       ["34.80%", f"{latest_rl['f1']*100:.2f}%"],
            }
            st.table(pd.DataFrame(comp_data))
    else:
        st.info("rl_training_log.json not found — run rl/ppo_train.py first.")