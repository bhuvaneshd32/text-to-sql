"""
grammar_fsm.py
--------------
SQL Grammar Finite State Machine + constrained decoding for Text-to-SQL.

WHAT THIS FILE DOES:
    Tracks exactly where we are in a SQL query as it is being generated
    token by token. At each decode step, returns a boolean mask over the
    full vocabulary — True = allowed, False = blocked.

    logits[~mask] = float("-inf")   # before softmax in decode loop
    next_token    = sample(logits)  # can ONLY be a valid token

API CONTRACT WITH NOOR:
    from grammar_fsm import get_mask, SQLGrammarFSM

    # In Noor's PPO decode loop:
    mask       = get_mask(partial_sql, db_id, tokenizer)
    logits[~mask] = float("-inf")
    next_token = Categorical(logits=logits).sample()

    Signature:
        get_mask(partial_sql: str, db_id: str, tokenizer) -> torch.BoolTensor
        Returns: bool tensor of shape [vocab_size]
                 True  = this token is allowed at the current state
                 False = this token is blocked

BUILD STRATEGY (do this in order, get each working before adding more):
    Phase 1: SELECT + FROM + WHERE       ← get this solid first
    Phase 2: JOIN
    Phase 3: GROUP BY + ORDER BY
    Phase 4: HAVING + LIMIT

WHY THIS REDUCES RL EXPLORATION:
    Without mask: ~50,265 tokens to explore per step
    With mask:    ~10–200 tokens depending on state
    Noor's PPO gradients are less noisy, training is more stable.
"""

import re
import torch
from enum import Enum, auto
from typing import Set, Dict, List, Optional
from functools import lru_cache


# ─────────────────────────────────────────────────────────────────
# FSM STATES
# ─────────────────────────────────────────────────────────────────

class SQLState(Enum):
    """
    Every state represents where we are in building the SQL query.
    The FSM is always in exactly one of these states.
    """
    START          = auto()   # beginning — only SELECT allowed
    SELECT_COLS    = auto()   # after SELECT — expect col names / * / agg
    SELECT_AGG     = auto()   # after COUNT/MAX/etc — expect (
    SELECT_AGG_COL = auto()   # inside COUNT( — expect col name or *
    AFTER_SELECT   = auto()   # after col list — expect FROM or comma
    FROM_TABLES    = auto()   # after FROM — expect table name
    AFTER_TABLE    = auto()   # after table name — WHERE/JOIN/GROUP/ORDER/HAVING/LIMIT/EOS
    JOIN_TABLE     = auto()   # after JOIN — expect table name
    JOIN_ON        = auto()   # after JOIN table — expect ON
    JOIN_COND_L    = auto()   # after ON — expect col name (left side)
    JOIN_COND_OP   = auto()   # after left col — expect =
    JOIN_COND_R    = auto()   # after = — expect col name (right side)
    WHERE_COL      = auto()   # after WHERE — expect col name
    WHERE_OP       = auto()   # after WHERE col — expect operator
    WHERE_VAL      = auto()   # after WHERE op — expect value
    WHERE_LOGIC    = auto()   # after WHERE val — expect AND/OR or clause keyword
    GROUP_COL      = auto()   # after GROUP BY — expect col name
    ORDER_COL      = auto()   # after ORDER BY — expect col name
    ORDER_DIR      = auto()   # after ORDER col — expect ASC/DESC
    HAVING_AGG     = auto()   # after HAVING — expect aggregation
    HAVING_PAREN   = auto()   # after HAVING agg — expect (
    HAVING_COL     = auto()   # inside HAVING agg( — expect col name
    HAVING_OP      = auto()   # after HAVING col — expect operator
    HAVING_VAL     = auto()   # after HAVING op — expect value
    LIMIT_VAL      = auto()   # after LIMIT — expect number
    DONE           = auto()   # after EOS — nothing allowed


# ─────────────────────────────────────────────────────────────────
# SQL KEYWORDS — pre-defined token sets
# ─────────────────────────────────────────────────────────────────

# Keywords we need to recognise when parsing partial_sql
AGGREGATIONS   = {"count", "max", "min", "avg", "sum"}
CLAUSE_KWS     = {"select", "from", "where", "group", "order",
                  "having", "limit", "join", "on", "and", "or",
                  "asc", "desc", "by", "distinct", "as",
                  "inner", "left", "right", "outer"}
OPERATORS      = {"=", "!=", "<>", ">", "<", ">=", "<=",
                  "like", "in", "not", "between", "is"}


class SQLGrammarFSM:
    """
    Finite State Machine for SQL grammar-constrained decoding.

    Usage:
        fsm = SQLGrammarFSM(db_id, schema_dict, tokenizer)
        mask = fsm.get_mask(partial_sql)   # [vocab_size] bool tensor
    """

    def __init__(
        self,
        db_id:       str,
        schema_dict: dict,
        tokenizer,
    ):
        """
        Args:
            db_id:       which database we are querying
            schema_dict: loaded from schema_utils.load_schema_dict()
            tokenizer:   HuggingFace AutoTokenizer (roberta-base)
        """
        self.db_id      = db_id
        self.tokenizer  = tokenizer
        self.vocab_size = tokenizer.vocab_size

        # ── extract schema info ───────────────────────────────────
        entry        = schema_dict[db_id]
        self.tables  = [t.lower() for t in entry["tables"]]
        self.columns = [c[0].lower() for c in entry["columns"]]

        # FIX: _tokenize_names now adds ALL subword tokens not just tokens[0]
        # so multi-token names like "Perpetrator_ID" are fully coverable
        self._table_ids   = self._tokenize_names(entry["tables"])
        self._column_ids  = self._tokenize_names([c[0] for c in entry["columns"]])
        self._agg_ids     = self._tokenize_keywords(
            ["COUNT", "MAX", "MIN", "AVG", "SUM",
             "count", "max", "min", "avg", "sum"]
        )
        self._op_ids      = self._tokenize_keywords(
            ["=", "!=", "<>", ">", "<", ">=", "<=",
             "LIKE", "IN", "NOT", "like", "in", "not"]
        )
        self._from_ids    = self._tokenize_keywords(["FROM", "from"])
        self._where_ids   = self._tokenize_keywords(["WHERE", "where"])
        self._join_ids    = self._tokenize_keywords(
            ["JOIN", "INNER", "LEFT", "RIGHT", "join", "inner", "left", "right"]
        )
        self._on_ids      = self._tokenize_keywords(["ON", "on"])
        self._and_or_ids  = self._tokenize_keywords(["AND", "OR", "and", "or"])
        self._group_ids   = self._tokenize_keywords(["GROUP", "group"])
        self._by_ids      = self._tokenize_keywords(["BY", "by", "ĠBY"])
        self._order_ids   = self._tokenize_keywords(["ORDER", "order"])
        self._having_ids  = self._tokenize_keywords(["HAVING", "having"])
        self._limit_ids   = self._tokenize_keywords(["LIMIT", "limit"])
        self._asc_desc_ids= self._tokenize_keywords(
            ["ASC", "DESC", "asc", "desc"]
        )
        self._comma_ids   = self._tokenize_keywords([","])
        self._star_ids    = self._tokenize_keywords(["*"])
        self._lparen_ids  = self._tokenize_keywords(["("])
        self._rparen_ids  = self._tokenize_keywords([")"])
        self._eq_ids      = self._tokenize_keywords(["="])
        self._eos_ids     = {tokenizer.eos_token_id} if tokenizer.eos_token_id else set()
        self._num_ids     = self._build_number_ids()

        # ── pre-build masks per state ─────────────────────────────
        # We build a boolean tensor for each FSM state once.
        # get_mask() just looks up the right one — very fast.
        self._state_masks: Dict[SQLState, torch.BoolTensor] = {}
        self._build_all_masks()

    # ── tokenization helpers ──────────────────────────────────────

    def _tokenize_names(self, names: List[str]) -> Set[int]:
        """
        Tokenize table/column names and collect token IDs.

        Same T5 SentencePiece fix as _tokenize_keywords: we look up the
        canonical surface forms (▁Name, ▁name, Name, name) via get_vocab()
        first, and fall back to subword splitting only when no exact match
        exists (needed for multi-token names like "Perpetrator_ID").
        """
        vocab = self.tokenizer.get_vocab()
        is_spm = any(k.startswith("▁") for k in list(vocab.keys())[:500])
        ids = set()

        for name in names:
            found_exact = False
            if is_spm:
                candidates = [
                    "▁" + name, "▁" + name.lower(), "▁" + name.upper(),
                    name, name.lower(), name.upper(),
                ]
            else:
                candidates = [
                    "Ġ" + name, "Ġ" + name.lower(), "Ġ" + name.upper(),
                    name, name.lower(), name.upper(),
                ]

            for candidate in candidates:
                if candidate in vocab:
                    ids.add(vocab[candidate])
                    found_exact = True

            if not found_exact:
                # Multi-token name — include all subword fragments so the
                # decoder can generate it piece by piece
                for text in [name, " " + name]:
                    toks = self.tokenizer(text, add_special_tokens=False)["input_ids"]
                    ids.update(toks)

        return ids

    def _tokenize_keywords(self, keywords: List[str]) -> Set[int]:
        """
        Tokenize SQL keywords and collect token IDs.

        T5 SentencePiece fix: encoding a multi-char keyword like "SELECT"
        produces subword fragments (e.g. [4248, 7, 4]) — none of which is the
        token the decoder actually samples (▁SELECT = token 3).

        We use get_vocab() to do an exact string lookup for the canonical
        surface forms a T5 decoder would generate:
            "▁SELECT"  (word-initial, SentencePiece space prefix)
            "SELECT"   (rare, no prefix)
            "▁select"  etc.
        and fall back to the subword split only when no exact match exists
        (needed for punctuation like "=", "(", ",").
        """
        vocab = self.tokenizer.get_vocab()
        ids = set()

        # Determine whether this is a SentencePiece tokenizer (T5) or BPE (RoBERTa)
        # SentencePiece uses "▁" (U+2581) as the word-boundary prefix.
        is_spm = any(k.startswith("▁") for k in list(vocab.keys())[:500])

        for kw in keywords:
            found_exact = False

            if is_spm:
                # Try all canonical surface forms T5 decoder would produce
                candidates = [
                    "▁" + kw,          # word-initial (most common)
                    "▁" + kw.lower(),
                    "▁" + kw.upper(),
                    kw,                # no prefix (mid-word or punctuation)
                    kw.lower(),
                    kw.upper(),
                ]
            else:
                # RoBERTa BPE: word-initial tokens are prefixed with Ġ
                candidates = [
                    "Ġ" + kw,
                    "Ġ" + kw.lower(),
                    "Ġ" + kw.upper(),
                    kw,
                    kw.lower(),
                    kw.upper(),
                ]

            for candidate in candidates:
                if candidate in vocab:
                    ids.add(vocab[candidate])
                    found_exact = True

            if not found_exact:
                # Fallback: subword split (needed for punctuation, numbers, etc.)
                for text in [kw, " " + kw]:
                    toks = self.tokenizer(text, add_special_tokens=False)["input_ids"]
                    ids.update(toks)

        return ids

    def _build_number_ids(self) -> Set[int]:
        """
        Build set of token IDs for numeric values.
        Pre-tokenize digits 0-9 and common numbers used in Spider.
        """
        ids = set()
        # digits and small numbers
        for n in list(range(0, 1000)) + [1000, 5000, 10000]:
            tokens = self.tokenizer(str(n), add_special_tokens=False)["input_ids"]
            ids.update(tokens)
            tokens_spaced = self.tokenizer(" " + str(n), add_special_tokens=False)["input_ids"]
            ids.update(tokens_spaced)
        for q in ['"', "'", '\u201c', '\u2018']:
            tokens = self.tokenizer(q, add_special_tokens=False)["input_ids"]
            ids.update(tokens)
        return ids

    # ── mask building ─────────────────────────────────────────────

    def _make_mask(self, allowed_ids: Set[int]) -> torch.BoolTensor:
        """
        Build a boolean tensor of shape [vocab_size].
        True at positions in allowed_ids, False everywhere else.
        """
        mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        for idx in allowed_ids:
            if 0 <= idx < self.vocab_size:
                mask[idx] = True
        return mask

    def _build_all_masks(self):
        """
        Pre-build one mask tensor per FSM state.
        Called once at __init__ — not at decode time.
        """
        # ── START: only SELECT allowed ────────────────────────────
        select_ids = self._tokenize_keywords(["SELECT", "select"])
        self._state_masks[SQLState.START] = self._make_mask(select_ids)

        # ── SELECT_COLS: col names, *, aggregations ───────────────
        self._state_masks[SQLState.SELECT_COLS] = self._make_mask(
            self._column_ids
            | self._star_ids
            | self._agg_ids
            | self._tokenize_keywords(["DISTINCT", "distinct"])
        )

        # ── SELECT_AGG: after COUNT/MAX etc, expect ( ─────────────
        self._state_masks[SQLState.SELECT_AGG] = self._make_mask(
            self._lparen_ids
        )

        # ── SELECT_AGG_COL: inside COUNT(, expect col name or * ───
        self._state_masks[SQLState.SELECT_AGG_COL] = self._make_mask(
            self._column_ids | self._star_ids
        )

        # ── AFTER_SELECT: ), comma, FROM ─────────────────────────
        self._state_masks[SQLState.AFTER_SELECT] = self._make_mask(
            self._rparen_ids
            | self._comma_ids
            | self._from_ids
        )

        # ── FROM_TABLES: only table names ────────────────────────
        self._state_masks[SQLState.FROM_TABLES] = self._make_mask(
            self._table_ids
        )

        # ── AFTER_TABLE: WHERE / JOIN / GROUP / ORDER / HAVING / LIMIT / EOS
        self._state_masks[SQLState.AFTER_TABLE] = self._make_mask(
            self._where_ids
            | self._join_ids
            | self._group_ids
            | self._order_ids
            | self._having_ids
            | self._limit_ids
            | self._eos_ids
            | self._comma_ids   # for subqueries
        )

        # ── JOIN_TABLE: table names ───────────────────────────────
        self._state_masks[SQLState.JOIN_TABLE] = self._make_mask(
            self._table_ids
        )

        # ── JOIN_ON: ON keyword ───────────────────────────────────
        self._state_masks[SQLState.JOIN_ON] = self._make_mask(
            self._on_ids
        )

        # ── JOIN_COND_L: col name (left of =) ────────────────────
        self._state_masks[SQLState.JOIN_COND_L] = self._make_mask(
            self._column_ids
        )

        # ── JOIN_COND_OP: = only ─────────────────────────────────
        self._state_masks[SQLState.JOIN_COND_OP] = self._make_mask(
            self._eq_ids
        )

        # ── JOIN_COND_R: col name (right of =) ───────────────────
        self._state_masks[SQLState.JOIN_COND_R] = self._make_mask(
            self._column_ids
        )

        # ── WHERE_COL: col names ──────────────────────────────────
        self._state_masks[SQLState.WHERE_COL] = self._make_mask(
            self._column_ids
        )

        # ── WHERE_OP: operators ───────────────────────────────────
        self._state_masks[SQLState.WHERE_OP] = self._make_mask(
            self._op_ids
        )

        # FIX: WHERE_VAL now includes exit tokens so episode can terminate
        # cleanly instead of looping until forced-EOS at t>25
        self._state_masks[SQLState.WHERE_VAL] = self._make_mask(
            self._num_ids
            | self._column_ids
            | self._and_or_ids
            | self._group_ids
            | self._order_ids
            | self._having_ids
            | self._limit_ids
            | self._eos_ids
        )

        self._state_masks[SQLState.WHERE_LOGIC] = self._make_mask(
            self._and_or_ids
            | self._group_ids
            | self._order_ids
            | self._having_ids
            | self._limit_ids
            | self._eos_ids
        )

        # ── GROUP_COL: col names, then BY ─────────────────────────
        self._state_masks[SQLState.GROUP_COL] = self._make_mask(
            self._column_ids | self._by_ids
        )

        # ── ORDER_COL: col names, then BY ─────────────────────────
        self._state_masks[SQLState.ORDER_COL] = self._make_mask(
            self._column_ids | self._by_ids
        )

        # ── ORDER_DIR: ASC / DESC / EOS ──────────────────────────
        self._state_masks[SQLState.ORDER_DIR] = self._make_mask(
            self._asc_desc_ids
            | self._limit_ids
            | self._eos_ids
        )

        # ── HAVING_AGG: aggregation functions ────────────────────
        self._state_masks[SQLState.HAVING_AGG] = self._make_mask(
            self._agg_ids
        )

        # ── HAVING_PAREN: ( ───────────────────────────────────────
        self._state_masks[SQLState.HAVING_PAREN] = self._make_mask(
            self._lparen_ids
        )

        # ── HAVING_COL: col name ──────────────────────────────────
        self._state_masks[SQLState.HAVING_COL] = self._make_mask(
            self._column_ids | self._star_ids
        )

        # ── HAVING_OP: operators ──────────────────────────────────
        self._state_masks[SQLState.HAVING_OP] = self._make_mask(
            self._op_ids
        )

        # ── HAVING_VAL: numbers ───────────────────────────────────
        self._state_masks[SQLState.HAVING_VAL] = self._make_mask(
            self._num_ids | self._rparen_ids
        )

        # ── LIMIT_VAL: number ─────────────────────────────────────
        self._state_masks[SQLState.LIMIT_VAL] = self._make_mask(
            self._num_ids
        )

        # ── DONE: nothing allowed ─────────────────────────────────
        self._state_masks[SQLState.DONE] = self._make_mask(set())

    # ── state inference ───────────────────────────────────────────

    def _infer_state(self, partial_sql: str) -> SQLState:
        """
        Walk through partial_sql tokens and determine current FSM state.

        Key fixes vs original:
        1. SELECT_COLS transitions to AFTER_SELECT on plain column token
           (original stayed in SELECT_COLS forever, blocking FROM)
        2. _in_agg flag disambiguates ) closing an aggregation vs spurious
           subword ) — prevents bouncing back to SELECT_COLS incorrectly
        3. WHERE_VAL transitions to DONE on unexpected CLAUSE_KWS
           so episode terminates instead of looping
        """
        if not partial_sql or not partial_sql.strip():
            return SQLState.START

        # normalise: lowercase, collapse whitespace
        sql = partial_sql.strip().lower()
        # split on whitespace and punctuation — keep punctuation as tokens
        tokens = re.findall(r"[\w.]+|[^\w\s]", sql)

        if not tokens:
            return SQLState.START

        state = SQLState.START
        _in_agg = False   # tracks whether we are inside COUNT(...) etc.

        i = 0
        while i < len(tokens):
            tok = tokens[i].lower()

            # ── transitions ───────────────────────────────────────
            if state == SQLState.START:
                if tok == "select":
                    state = SQLState.SELECT_COLS

            elif state == SQLState.SELECT_COLS:
                if tok in AGGREGATIONS:
                    state = SQLState.SELECT_AGG
                    _in_agg = True
                elif tok == "from":
                    state = SQLState.FROM_TABLES
                elif tok == ",":
                    state = SQLState.SELECT_COLS
                elif tok == "distinct":
                    state = SQLState.SELECT_COLS   # stay, column follows
                elif tok == "*":
                    # FIX: * is a complete column reference — expect FROM or comma
                    state = SQLState.AFTER_SELECT
                elif tok not in CLAUSE_KWS:
                    # FIX: plain column name seen — transition to AFTER_SELECT
                    # Original stayed in SELECT_COLS here which blocked FROM forever
                    state = SQLState.AFTER_SELECT

            elif state == SQLState.SELECT_AGG:
                if tok == "(":
                    state = SQLState.SELECT_AGG_COL

            elif state == SQLState.SELECT_AGG_COL:
                # * or col name — move to AFTER_SELECT for )
                if tok not in {"("}:
                    state = SQLState.AFTER_SELECT

            elif state == SQLState.AFTER_SELECT:
                if tok == ")" and _in_agg:
                    # FIX: ) closes the aggregation paren — back to col list
                    # _in_agg flag prevents spurious ) from resetting state
                    state   = SQLState.SELECT_COLS
                    _in_agg = False
                elif tok == ")":
                    # spurious subword ) — stay in AFTER_SELECT, allow FROM
                    pass
                elif tok == ",":
                    state = SQLState.SELECT_COLS
                elif tok == "from":
                    state = SQLState.FROM_TABLES

            elif state == SQLState.FROM_TABLES:
                if tok not in CLAUSE_KWS:
                    # table name — move to AFTER_TABLE
                    state = SQLState.AFTER_TABLE
                elif tok == "where":
                    state = SQLState.WHERE_COL
                elif tok in {"join", "inner", "left", "right"}:
                    state = SQLState.JOIN_TABLE

            elif state == SQLState.AFTER_TABLE:
                if tok == "where":
                    state = SQLState.WHERE_COL
                elif tok in {"join", "inner", "left", "right"}:
                    state = SQLState.JOIN_TABLE
                elif tok == "group":
                    state = SQLState.GROUP_COL
                elif tok == "order":
                    state = SQLState.ORDER_COL
                elif tok == "having":
                    state = SQLState.HAVING_AGG
                elif tok == "limit":
                    state = SQLState.LIMIT_VAL

            elif state == SQLState.JOIN_TABLE:
                if tok not in CLAUSE_KWS:
                    state = SQLState.JOIN_ON

            elif state == SQLState.JOIN_ON:
                if tok == "on":
                    state = SQLState.JOIN_COND_L

            elif state == SQLState.JOIN_COND_L:
                if tok not in CLAUSE_KWS and tok != "=":
                    state = SQLState.JOIN_COND_OP

            elif state == SQLState.JOIN_COND_OP:
                if tok == "=":
                    state = SQLState.JOIN_COND_R

            elif state == SQLState.JOIN_COND_R:
                if tok not in CLAUSE_KWS:
                    state = SQLState.AFTER_TABLE

            elif state == SQLState.WHERE_COL:
                if tok not in CLAUSE_KWS and tok not in OPERATORS:
                    state = SQLState.WHERE_OP

            elif state == SQLState.WHERE_OP:
                if tok in OPERATORS:
                    state = SQLState.WHERE_VAL

            elif state == SQLState.WHERE_VAL:
                if tok in {"and", "or"}:
                    state = SQLState.WHERE_COL
                elif tok == "group":
                    state = SQLState.GROUP_COL
                elif tok == "order":
                    state = SQLState.ORDER_COL
                elif tok == "having":
                    state = SQLState.HAVING_AGG
                elif tok == "limit":
                    state = SQLState.LIMIT_VAL
                elif tok in CLAUSE_KWS - {
                    "and", "or", "group", "order", "having",
                    "limit", "not", "like", "in", "between", "is"
                }:
                    # FIX: unexpected clause keyword means we overshot —
                    # terminate cleanly rather than looping in WHERE_VAL
                    state = SQLState.DONE
                # else: stay in WHERE_VAL (multi-token value)

            elif state == SQLState.GROUP_COL:
                if tok == "by":
                    pass
                elif tok == "order":
                    state = SQLState.ORDER_COL
                elif tok == "having":
                    state = SQLState.HAVING_AGG
                elif tok == "limit":
                    state = SQLState.LIMIT_VAL
                elif tok not in CLAUSE_KWS:
                    state = SQLState.AFTER_TABLE

            elif state == SQLState.ORDER_COL:
                if tok == "by":
                    pass
                elif tok not in CLAUSE_KWS:
                    state = SQLState.ORDER_DIR

            elif state == SQLState.ORDER_DIR:
                if tok in {"asc", "desc"}:
                    state = SQLState.AFTER_TABLE
                elif tok == "limit":
                    state = SQLState.LIMIT_VAL

            elif state == SQLState.HAVING_AGG:
                if tok in AGGREGATIONS:
                    state = SQLState.HAVING_PAREN

            elif state == SQLState.HAVING_PAREN:
                if tok == "(":
                    state = SQLState.HAVING_COL

            elif state == SQLState.HAVING_COL:
                if tok not in {"(", ")"}:
                    state = SQLState.HAVING_OP

            elif state == SQLState.HAVING_OP:
                if tok in OPERATORS:
                    state = SQLState.HAVING_VAL

            elif state == SQLState.HAVING_VAL:
                if tok == ")":
                    state = SQLState.AFTER_TABLE
                elif tok not in CLAUSE_KWS:
                    pass

            elif state == SQLState.LIMIT_VAL:
                if tok.isdigit():
                    state = SQLState.DONE

            i += 1

        return state

    def get_mask(self, partial_sql: str) -> torch.BoolTensor:
        """
        Get the valid token mask for the current decode position.

        Args:
            partial_sql: SQL string generated so far (may be empty)

        Returns:
            BoolTensor [vocab_size] — True = valid token at this position
        """
        state = self._infer_state(partial_sql)
        mask  = self._state_masks.get(state)

        # fallback: allow everything if state has no mask (shouldn't happen)
        if mask is None:
            return torch.ones(self.vocab_size, dtype=torch.bool)

        return mask.clone()


# ─────────────────────────────────────────────────────────────────
# MODULE-LEVEL get_mask FUNCTION (API contract with Noor)
# ─────────────────────────────────────────────────────────────────

# Cache one FSM per (db_id) to avoid re-building at every decode step
_fsm_cache: Dict[str, SQLGrammarFSM] = {}
_schema_dict_cache = None

def get_mask(
    partial_sql: str,
    db_id:       str,
    tokenizer,
    schema_dict: dict = None,
) -> torch.BoolTensor:
    """
    Module-level convenience function — API contract with Noor.

    Signature (agreed on Day 5):
        get_mask(partial_sql, db_id, tokenizer) -> BoolTensor [vocab_size]

    Caches one FSM per db_id so it is only built once per episode.

    Usage in Noor's PPO decode loop:
        from grammar_fsm import get_mask
        mask = get_mask(partial_sql, db_id, tokenizer)
        logits[~mask] = float("-inf")
        next_token = Categorical(logits=logits).sample()
    """
    global _fsm_cache, _schema_dict_cache

    if schema_dict is not None:
        _schema_dict_cache = schema_dict

    if db_id not in _fsm_cache:
        if _schema_dict_cache is None:
            raise RuntimeError(
                "schema_dict must be passed on the first call to get_mask() "
                "for each new db_id."
            )
        _fsm_cache[db_id] = SQLGrammarFSM(db_id, _schema_dict_cache, tokenizer)

    return _fsm_cache[db_id].get_mask(partial_sql)


# ─────────────────────────────────────────────────────────────────
# VERIFICATION TEST
# Run: python grammar_fsm.py
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from transformers import AutoTokenizer
    from config import TABLES_JSON
    from schema_utils import load_schema_dict

    print("=" * 60)
    print("Task 5 — Grammar FSM Verification")
    print("=" * 60)

    tokenizer   = AutoTokenizer.from_pretrained("roberta-base")
    schema_dict = load_schema_dict(TABLES_JSON)
    db_id       = "perpetrator"

    fsm = SQLGrammarFSM(db_id, schema_dict, tokenizer)

    print(f"\nDatabase: {db_id}")
    print(f"Tables:   {schema_dict[db_id]['tables']}")
    print(f"Vocab size: {tokenizer.vocab_size}")

    # ── test state inference ──────────────────────────────────────
    print("\n--- State inference tests ---")
    test_cases = [
        ("",                                        SQLState.START),
        ("SELECT",                                  SQLState.SELECT_COLS),
        ("SELECT count",                            SQLState.SELECT_AGG),
        ("SELECT count (",                          SQLState.SELECT_AGG_COL),
        ("SELECT count ( *",                        SQLState.AFTER_SELECT),
        ("SELECT count ( * )",                      SQLState.SELECT_COLS),
        ("SELECT count ( * ) FROM",                 SQLState.FROM_TABLES),
        ("SELECT count ( * ) FROM perpetrator",     SQLState.AFTER_TABLE),
        ("SELECT name",                             SQLState.AFTER_SELECT),   # FIX test
        ("SELECT name FROM",                        SQLState.FROM_TABLES),    # FIX test
        ("SELECT * FROM perpetrator WHERE",         SQLState.WHERE_COL),
        ("SELECT * FROM perpetrator WHERE Killed",  SQLState.WHERE_OP),
        ("SELECT * FROM perpetrator WHERE Killed >",SQLState.WHERE_VAL),
        ("SELECT * FROM perpetrator JOIN",          SQLState.JOIN_TABLE),
        ("SELECT * FROM perpetrator GROUP",         SQLState.GROUP_COL),
        ("SELECT * FROM perpetrator ORDER",         SQLState.ORDER_COL),
    ]

    all_pass = True
    for partial, expected in test_cases:
        got = fsm._infer_state(partial)
        status = "✓" if got == expected else "✗"
        if got != expected:
            all_pass = False
        print(f"  {status} '{partial[:50]:<50}' → {got.name}")

    print("\n✓ All passed." if all_pass else "\n✗ Some failed — check transitions.")

    print("\n--- Action space reduction ---")
    checks = [
        ("",                                     "START"),
        ("SELECT",                               "SELECT_COLS"),
        ("SELECT name",                          "AFTER_SELECT"),
        ("SELECT count ( * ) FROM",              "FROM_TABLES"),
        ("SELECT * FROM perpetrator WHERE",      "WHERE_COL"),
        ("SELECT * FROM perpetrator WHERE Killed", "WHERE_OP"),
    ]
    print(f"  {'Partial SQL':<50} {'State':<15} {'Valid tokens'}")
    print(f"  {'-'*80}")
    for partial, sname in checks:
        mask    = fsm.get_mask(partial)
        n_valid = mask.sum().item()
        pct     = 100 * n_valid / tokenizer.vocab_size
        print(f"  {partial[:49]:<50} {sname:<15} {n_valid:5d}  ({pct:.2f}%)")

    print(f"\n  Full vocab: {tokenizer.vocab_size}")
    print(f"\n✓ grammar_fsm.py verified and ready.")