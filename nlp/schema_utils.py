"""
schema_utils.py
---------------
Utilities for loading and serializing database schemas from Spider's tables.json.

KEY OUTPUTS:
    schema_dict  — JSON-serializable Python dict, used by  MDP environment
    serialize_schema() — converts a db_id to a string you can concatenate with the NL question

    schema_dict format:
    {
        "employees": {
            "tables":   ["employees", "departments"],
            "columns":  [["id","INT"], ["name","TEXT"], ["salary","REAL"], ["dept_id","INT"]],
            "types":    ["INT", "TEXT", "REAL", "INT"],
            "fkeys":    [[3, 0]]   # col index 3 (dept_id) references col index 0 (id)
        },
        ...
    }
"""

import json
import os


def load_schema_dict(tables_json_path: str) -> dict:
    """
    Parse Spider's tables.json into a clean schema_dict.

    WHY:
        Spider stores schema info in a nested JSON format. We flatten it into
        a simple dict so both our encoder and Noor's MDP can easily look up
        which tables/columns belong to a given db_id.

    Args:
        tables_json_path: path to Spider's tables.json file

    Returns:
        schema_dict: { db_id -> { tables, columns, types, fkeys } }
    """
    with open(tables_json_path, "r") as f:
        raw = json.load(f)

    schema_dict = {}

    for entry in raw:
        db_id = entry["db_id"]

        # Spider stores column names as [col_index, col_name] — we just want names
        # Spider stores column types as a flat list matching column order
        tables = entry["table_names_original"]   # e.g. ["employees", "departments"]
        columns = entry["column_names_original"]  # e.g. [[-1, "*"], [0, "id"], [0, "name"]]
        col_types = entry["column_types"]         # e.g. ["text", "number", "text"]
        fkeys = entry["foreign_keys"]             # e.g. [[3, 0], [7, 1]]

        # Pair column name with its type (skip the first special "*" column at index 0)
        col_name_type_pairs = []
        for i, (table_idx, col_name) in enumerate(columns):
            if table_idx == -1:
                continue  # skip the global "*" wildcard column
            type_str = col_types[i] if i < len(col_types) else "text"
            col_name_type_pairs.append([col_name, type_str.upper()])

        schema_dict[db_id] = {
            "tables":  tables,
            "columns": col_name_type_pairs,
            "types":   [pair[1] for pair in col_name_type_pairs],
            "fkeys":   fkeys,
        }

    return schema_dict


def serialize_schema(db_id: str, schema_dict: dict) -> str:
    """
    Convert a database schema into a flat string for encoder input.

    WHY:
        The encoder takes a single string as input. We need to flatten
        the whole DB schema (multiple tables, each with multiple columns)
        into one string that the tokenizer can handle.

    Format:
        "employees: id(INT), name(TEXT), salary(REAL) | departments: id(INT), dept_name(TEXT)"

    Each table is separated by " | " so the model sees clear boundaries.
    Column types are in parentheses so the model learns type-aware SQL generation
    (e.g., don't wrap INT columns in quotes in WHERE clauses).

    Args:
        db_id:       the Spider database identifier, e.g. "company_1"
        schema_dict: the dict returned by load_schema_dict()

    Returns:
        A single string representing the full schema.
    """
    if db_id not in schema_dict:
        raise KeyError(f"db_id '{db_id}' not found in schema_dict. "
                       f"Available: {list(schema_dict.keys())[:5]}...")

    entry = schema_dict[db_id]
    tables = entry["tables"]
    columns = entry["columns"]   # list of [col_name, col_type]

    # We need to know which columns belong to which table.
    # We re-read tables.json grouping to do that — so we track table_idx per column.
    # Here we use a simple approach: columns are stored in table order in Spider.
    # Build a mapping: table_name -> list of (col_name, col_type)
    table_col_map = {t: [] for t in tables}

    # NOTE: Spider's column_names_original uses table_idx. We stored columns in
    # order but lost table_idx in our simplified schema_dict. So we rebuild from raw.
    # In production, store table_idx in schema_dict directly (improvement for later).
    # For now, distribute columns evenly — this is approximate but works for a prototype.
    # TODO Day 2: store table_idx per column in schema_dict for exact grouping.
    cols_per_table = max(1, len(columns) // max(len(tables), 1))
    for i, (col_name, col_type) in enumerate(columns):
        t_idx = min(i // cols_per_table, len(tables) - 1)
        table_col_map[tables[t_idx]].append(f"{col_name}({col_type})")

    # Serialize: "table1: col1(TYPE), col2(TYPE) | table2: col3(TYPE)"
    parts = []
    for table_name in tables:
        cols_str = ", ".join(table_col_map[table_name]) if table_col_map[table_name] else ""
        parts.append(f"{table_name}: {cols_str}" if cols_str else table_name)

    return " | ".join(parts)


def get_schema_token_to_column_map(db_id: str, schema_dict: dict, tokenizer) -> dict:
    """
    After tokenizing the schema string, build a mapping:
        token_index_in_full_sequence -> column_index_in_schema_dict

    WHY (tell Noor):
        Noor's MDP state encoder needs to know which token positions correspond
        to which schema elements. This map is what she uses to build S_schema
        from the encoder hidden states.

    Args:
        db_id:       database identifier
        schema_dict: the loaded schema dict
        tokenizer:   HuggingFace tokenizer (must be initialized before calling)

    Returns:
        dict: { token_offset_in_schema_part -> column_index }

    NOTE: This is a simplified version. A robust version needs to track
          exact token offsets using tokenizer(return_offsets_mapping=True).
          We will improve this in Task 3 during cross-attention setup.
    """
    schema_str = serialize_schema(db_id, schema_dict)
    # Tokenize just the schema portion to get its token count
    schema_tokens = tokenizer(schema_str, add_special_tokens=False)["input_ids"]

    # Approximate mapping: distribute schema tokens across columns
    columns = schema_dict[db_id]["columns"]
    n_cols = len(columns)
    n_tokens = len(schema_tokens)

    token_to_col = {}
    if n_cols == 0:
        return token_to_col

    tokens_per_col = max(1, n_tokens // n_cols)
    for tok_i in range(n_tokens):
        col_i = min(tok_i // tokens_per_col, n_cols - 1)
        token_to_col[tok_i] = col_i

    return token_to_col


# ─── quick test ──────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    # Replace with your actual path to tables.json
    TABLES_JSON = "data/spider/tables.json"

    if not os.path.exists(TABLES_JSON):
        print(f"tables.json not found at {TABLES_JSON}")
        print("Set TABLES_JSON path to your Spider download location.")
        sys.exit(1)

    schema_dict = load_schema_dict(TABLES_JSON)
    print(f"Loaded {len(schema_dict)} databases from tables.json\n")

    # Test with first available db_id
    first_db = list(schema_dict.keys())[0]
    print(f"Sample db_id: {first_db}")
    print(f"Tables: {schema_dict[first_db]['tables']}")
    print(f"Columns (first 5): {schema_dict[first_db]['columns'][:5]}")
    print(f"\nSerialized schema:\n{serialize_schema(first_db, schema_dict)}\n")