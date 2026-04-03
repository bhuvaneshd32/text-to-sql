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


def efficiency_reward(sql_tokens, max_len=128):
    return -(len(sql_tokens) / max_len)


def compute_reward(
    pred_sql,
    gold_sql,
    db_path,
    sql_tokens,
    alpha=1.0,
    beta=0.5,
    gamma=0.1,
):
    """
    Combined reward:
        R_total = alpha * R_exec + beta * R_sem + gamma * R_eff

    Uses Bhuvanesh's shared metrics directly so reward signal
    matches evaluation metrics exactly.
    """
    R_exec = float(exec_accuracy(pred_sql, gold_sql, db_path))
    R_sem  = result_set_f1(pred_sql, gold_sql, db_path)
    R_eff  = efficiency_reward(sql_tokens)

    return alpha * R_exec + beta * R_sem + gamma * R_eff