# # rl/reward.py
# import sqlite3


# # ─────────────────────────────────────────────
# # SQL EXECUTION
# # ─────────────────────────────────────────────
# def execute_sql(db_path, query):
#     """
#     Execute SQL query on SQLite DB.

#     Returns:
#         result (list of tuples) or None
#         success (bool)
#     """
#     try:
#         conn = sqlite3.connect(db_path)
#         cursor = conn.cursor()
#         cursor.execute(query)
#         result = cursor.fetchall()
#         conn.close()
#         return result, True
#     except Exception:
#         return None, False


# # ─────────────────────────────────────────────
# # EXECUTION REWARD (R_exec)
# # ─────────────────────────────────────────────
# def execution_reward(pred_sql, gold_sql, db_path):
#     pred_res, pred_ok = execute_sql(db_path, pred_sql)
#     gold_res, gold_ok = execute_sql(db_path, gold_sql)

#     if not pred_ok:
#         return 0.0

#     if pred_res == gold_res:
#         return 1.0

#     return 0.0


# # ─────────────────────────────────────────────
# # SEMANTIC REWARD (R_sem) — Result F1
# # ─────────────────────────────────────────────
# def result_f1(pred_res, gold_res):
#     pred_set = set(pred_res) if pred_res else set()
#     gold_set = set(gold_res) if gold_res else set()

#     if len(pred_set) == 0 and len(gold_set) == 0:
#         return 1.0

#     if len(pred_set) == 0 or len(gold_set) == 0:
#         return 0.0

#     intersection = len(pred_set & gold_set)

#     precision = intersection / len(pred_set)
#     recall    = intersection / len(gold_set)

#     if precision + recall == 0:
#         return 0.0

#     return 2 * precision * recall / (precision + recall)


# # ─────────────────────────────────────────────
# # EFFICIENCY REWARD (R_eff)
# # ─────────────────────────────────────────────
# def efficiency_reward(sql_tokens, max_len=128):
#     """
#     Penalize long SQL queries.
#     """
#     length_penalty = len(sql_tokens) / max_len
#     return -length_penalty


# # ─────────────────────────────────────────────
# # TOTAL REWARD
# # ─────────────────────────────────────────────
# def compute_reward(
#     pred_sql,
#     gold_sql,
#     db_path,
#     sql_tokens,
#     alpha=1.0,
#     beta=0.5,
#     gamma=0.1,
# ):
#     """
#     Combined reward:
#         R_total = alpha * R_exec + beta * R_sem + gamma * R_eff
#     """

#     pred_res, pred_ok = execute_sql(db_path, pred_sql)
#     gold_res, _       = execute_sql(db_path, gold_sql)

#     R_exec = 1.0 if pred_ok and pred_res == gold_res else 0.0
#     R_sem  = result_f1(pred_res or [], gold_res or [])
#     R_eff  = efficiency_reward(sql_tokens)

#     return alpha * R_exec + beta * R_sem + gamma * R_eff

# rl/reward.py
from nlp.eval_utils import exec_accuracy, result_set_f1
import re


def efficiency_reward(sql_tokens, max_len=128):
    """Only penalize queries significantly over half max length."""
    length = len(sql_tokens)
    if length <= max_len * 0.5:
        return 0.0
    return -((length - max_len * 0.5) / (max_len * 0.5))


def counterfactual_reward(pred_sql, gold_sql, db_path):
    """
    R3 — Counterfactual Consistency.
    Checks aggregation and LIMIT alignment between pred and gold.
    Returns float in [-0.05, 0.05]
    """
    from nlp.eval_utils import execute_sql as exec_sql

    pred_result = exec_sql(pred_sql, db_path)
    gold_result = exec_sql(gold_sql, db_path)

    if pred_result is None:
        return -0.05
    if gold_result is None:
        return 0.0

    gold_lower = gold_sql.lower()
    pred_lower = pred_sql.lower()
    aggs = ["count", "avg", "sum", "max", "min"]

    gold_agg = [a for a in aggs if a in gold_lower]
    pred_agg = [a for a in aggs if a in pred_lower]

    if gold_agg and not pred_agg:
        return -0.05
    if not gold_agg and pred_agg:
        return -0.03

    gold_limit = re.search(r'limit\s+(\d+)', gold_lower)
    pred_limit = re.search(r'limit\s+(\d+)', pred_lower)

    if gold_limit and pred_limit:
        if gold_limit.group(1) == pred_limit.group(1):
            return 0.05
    elif gold_limit and not pred_limit:
        return -0.03

    return 0.0


def compute_reward(
    pred_sql,
    gold_sql,
    db_path,
    sql_tokens,
    alpha=1.0,
    beta=0.5,
    gamma=0.1,
    delta=0.3,
):
    """
    R_total = alpha*R_exec + beta*R_sem + gamma*R_eff + delta*R_cf
    """
    R_exec = float(exec_accuracy(pred_sql, gold_sql, db_path))
    R_sem  = result_set_f1(pred_sql, gold_sql, db_path)
    R_eff  = efficiency_reward(sql_tokens)
    R_cf   = counterfactual_reward(pred_sql, gold_sql, db_path)

    return alpha * R_exec + beta * R_sem + gamma * R_eff + delta * R_cf