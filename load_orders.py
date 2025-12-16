#!/usr/bin/env python3
"""
Load orders Excel/CSV into sqlite using ShipCube client_orders schema.

Usage:
    python load_orders.py path/to/orders.xlsx
"""

import sys
import os
import sqlite3
import pandas as pd
import json
import re
from pathlib import Path
from tqdm import tqdm

DB_PATH = Path("data/shipcube.db")
TABLE_NAME = "client_orders"
MAPPING_PATH = Path("data/col_map.json")
CHUNK = 1000


# ------------------ Helpers ------------------

def to_snake(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    s = s.replace("\n", " ").replace("\r", " ")
    s = re.sub(r"[^0-9A-Za-z]+", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_").lower()
    if not re.match(r"^[a-z]", s):
        s = "c_" + s
    return s

def normalize_value(val):
    """Convert pandas / numpy values into SQLite-safe types."""
    if pd.isna(val):
        return None

    # pandas Timestamp or datetime
    if isinstance(val, (pd.Timestamp,)):
        return val.isoformat()

    return val

# ------------------ Canonical schema ------------------

CANONICAL_COLUMNS = {
    "shipping_label_id": ["shipping_label_id", "label_id"],
    "order_date": ["order_date", "order date"],
    "order_number": ["order_number", "order no", "order number"],
    "quantity_shipped": ["quantity_shipped", "qty shipped", "quantity"],
    "order_id": ["order_id"],
    "carrier": ["carrier"],
    "shipping_method": ["shipping_method", "service", "method"],
    "tracking_number": ["tracking_number", "tracking no", "tracking"],
    "created_at": ["created_at", "created date"],
    "to_name": ["to_name", "customer", "recipient"],
    "final_amount": ["final_amount", "amount", "invoice amount", "total"],
    "zip": ["zip", "zipcode", "postal"],
    "state": ["state"],
    "country": ["country"],
    "size_dimensions": ["size", "dimensions"],
    "weight_oz": ["weight", "weight_oz"],
    "tpl_customer": ["3pl customer", "tpl customer", "client"],
    "warehouse": ["warehouse"]
}


# ------------------ DB Schema ------------------

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS client_orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    shipping_label_id TEXT,
    order_date TEXT,
    order_number TEXT,
    quantity_shipped INTEGER,
    order_id TEXT,
    carrier TEXT,
    shipping_method TEXT,
    tracking_number TEXT,
    created_at TEXT,
    to_name TEXT,
    final_amount REAL,
    zip TEXT,
    state TEXT,
    country TEXT,
    size_dimensions TEXT,
    weight_oz REAL,
    tpl_customer TEXT,
    warehouse TEXT
)
"""


# ------------------ Core Loader ------------------

def dataframe_to_sqlite(df: pd.DataFrame):
    os.makedirs(DB_PATH.parent, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Normalize dataframe columns
    orig_cols = list(df.columns)
    norm_cols = {c: to_snake(c) for c in orig_cols}

    # Build mapping → canonical schema
    resolved_map = {}
    for canonical, variants in CANONICAL_COLUMNS.items():
        for col, norm in norm_cols.items():
            if norm in [to_snake(v) for v in variants]:
                resolved_map[canonical] = col
                break

    # Recreate table
    cur.execute("DROP TABLE IF EXISTS client_orders")
    cur.execute(CREATE_TABLE_SQL)
    conn.commit()

    insert_cols = list(resolved_map.keys())
    placeholders = ", ".join(["?"] * len(insert_cols))
    insert_sql = f"""
        INSERT INTO client_orders ({", ".join(insert_cols)})
        VALUES ({placeholders})
    """

    total = len(df)
    print(f"Inserting {total} rows into client_orders...")

    batch = []
    for _, row in tqdm(df.iterrows(), total=total):
        vals = []
        for col in insert_cols:
            src = resolved_map.get(col)
            raw = row[src] if src else None
            vals.append(normalize_value(raw))

        batch.append(vals)

        if len(batch) >= CHUNK:
            cur.executemany(insert_sql, batch)
            conn.commit()
            batch = []

    if batch:
        cur.executemany(insert_sql, batch)
        conn.commit()

    conn.close()

    # Save mapping
    with open(MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump(resolved_map, f, indent=2)

    print(" Load complete")
    print(" Column mapping saved to:", MAPPING_PATH)


# ------------------ Main ------------------

def main(argv):
    if len(argv) < 2:
        print("Usage: python load_orders.py path/to/orders.xlsx")
        return 1

    path = Path(argv[1])
    if not path.exists():
        print("File not found:", path)
        return 1

    print("Reading:", path)
    if path.suffix.lower() in [".xls", ".xlsx"]:
        df = pd.read_excel(path, engine="openpyxl")
    else:
        df = pd.read_csv(path)

    print("Detected columns:", list(df.columns))
    dataframe_to_sqlite(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
