#!/usr/bin/env python3
"""
Load orders Excel/CSV into sqlite and normalize column names.

Usage:
    python load_orders.py path/to/orders.xlsx

Creates:
  - data/shipcube.db (if not existing)
  - table: client_orders (recreated)
  - data/col_map.json (original -> normalized mapping)
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
CHUNK = 1000  # commit every N rows

def to_snake(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    s = s.replace('\n',' ').replace('\r',' ')
    # common replacements
    s = s.replace('(UTC)','').replace('(Added 50 Cents)','')
    s = re.sub(r'[^0-9A-Za-z]+', '_', s)           # anything non-alnum -> underscore
    s = re.sub(r'_{2,}', '_', s)                   # collapse multiple underscores
    s = s.strip('_').lower()
    # ensure starts with letter
    if not re.match(r'^[a-z]', s):
        s = 'c_' + s
    return s

def dataframe_to_sqlite(df: pd.DataFrame, db_path: Path, table_name: str):
    os.makedirs(db_path.parent, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()

    # Normalize columns
    orig_cols = list(df.columns)
    normalized = [to_snake(c) for c in orig_cols]

    # If duplicate normalized names occur, make them unique by appending index
    seen = {}
    final_cols = []
    for name in normalized:
        base = name
        idx = 1
        while name in seen:
            idx += 1
            name = f"{base}_{idx}"
        seen[name] = True
        final_cols.append(name)

    # Build mapping
    col_map = {orig: norm for orig, norm in zip(orig_cols, final_cols)}

    # Drop table if exists (we recreate clean)
    cur.execute(f'DROP TABLE IF EXISTS "{table_name}"')

    # Create table with TEXT columns
    col_defs = ", ".join([f'"{col}" TEXT' for col in final_cols])
    create_sql = f'CREATE TABLE "{table_name}" (id INTEGER PRIMARY KEY AUTOINCREMENT, {col_defs})'
    cur.execute(create_sql)
    con.commit()
    # Build insert SQL safely
    col_list = ", ".join([f'"{c}"' for c in final_cols])
    placeholders = ", ".join(["?"] * len(final_cols))
    insert_sql = f'INSERT INTO "{table_name}" ({col_list}) VALUES ({placeholders})'

    total = len(df)
    print(f"Inserting {total} rows into {db_path}/{table_name} ...")
    rows = df.fillna("").astype(str)
    batch = []
    count = 0
    for idx, row in tqdm(rows.iterrows(), total=total):
        vals = [row[orig_cols[i]] for i in range(len(orig_cols))]
        batch.append(vals)
        if len(batch) >= CHUNK:
            cur.executemany(insert_sql, batch)
            con.commit()
            count += len(batch)
            batch = []
    if batch:
        cur.executemany(insert_sql, batch)
        con.commit()
        count += len(batch)

    print(f"Inserted {count} rows.")

    # Create indexes on likely lookup fields if they exist in final_cols
    # common names we will look for (converted to snake)
    candidates = {
        "order_id": ["order id","orderid","order_id","order_id_1","c_order_id"],
        "store_order_id": ["store orderid","storeorderid","store_orderid","store_order_id"],
        "tracking_id": ["tracking id","trackingid","tracking_id"],
        "invoice_number": ["invoice number","invoice_number","invoice"],
        "merchant_name": ["merchant name","merchant_name","merchant"]
    }
    created_indexes = []
    for logical, variants in candidates.items():
        for var in variants:
            norm = to_snake(var)
            # if mapping produced a column matching the variant, use it
            # but better: check final_cols for substring or exact match
            matched = None
            # exact first
            if norm in final_cols:
                matched = norm
            else:
                # fuzzy: look for any final_col that contains tokens
                tokens = [t for t in re.split(r'[_\s]+', norm) if t]
                for fc in final_cols:
                    if all(tok in fc for tok in tokens):
                        matched = fc
                        break
            if matched:
                idx_name = f"idx_{table_name}_{matched}"
                try:
                    cur.execute(f'CREATE INDEX IF NOT EXISTS "{idx_name}" ON "{table_name}"("{matched}")')
                    created_indexes.append(matched)
                except Exception as e:
                    print("Index creation failed for", matched, e)
                break

    con.commit()
    con.close()

    # Save mapping
    with open(MAPPING_PATH, "w", encoding="utf-8") as fh:
        json.dump({"mapping": col_map, "created_indexes": created_indexes}, fh, indent=2)

    print("Column mapping saved to", MAPPING_PATH)
    print("Indexes created on:", created_indexes)

def main(argv):
    if len(argv) < 2:
        print("Usage: python load_orders.py path/to/orders.xlsx")
        return 1
    path = Path(argv[1])
    if not path.exists():
        print("File not found:", path)
        return 1

    # read with pandas (auto detects xlsx/csv)
    print("Reading file:", path)
    if path.suffix.lower() in [".xls", ".xlsx"]:
        df = pd.read_excel(str(path), engine="openpyxl")
    else:
        df = pd.read_csv(str(path))

    print("Columns found:", list(df.columns)[:20])
    dataframe_to_sqlite(df, DB_PATH, TABLE_NAME)
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
