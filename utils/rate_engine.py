# utils/rate_engine.py

import pandas as pd
import numpy as np
import os
from typing import Dict, List

# ---------------- CONFIG ---------------- #

DATA_DIR = "data/rates"
ZONE_FILE = f"{DATA_DIR}/zone_lookup.csv"

RATE_FILES = {
    "IB": f"{DATA_DIR}/IB.csv",
    "national_carrier": f"{DATA_DIR}/national_carrier.csv",
    "regional_carr": f"{DATA_DIR}/regional_carr.csv",
    "shipbob": f"{DATA_DIR}/shipbob.csv",
    "speedx": f"{DATA_DIR}/speedx.csv",
    "Tier4": f"{DATA_DIR}/tier4.csv",
    "Uniuni5000": f"{DATA_DIR}/uniuni5000.csv",
    "Veho": f"{DATA_DIR}/veho.csv",
    "FMGN": f"{DATA_DIR}/FMGN.csv",
    "FMGS": f"{DATA_DIR}/FMGS.csv",
    "JITSU": f"{DATA_DIR}/JITSU.csv",
}

ZONE_PREFIX_COL = "Zipcode_Prefix"
RATE_WEIGHT_COL = "Weight_Max"

# ---------------- GLOBAL STORAGE ---------------- #

ZONE_DATA: Dict[int, Dict[str, int]] = {}
RATES_DATA: Dict[str, pd.DataFrame] = {}

# ---------------- LOAD DATA ---------------- #

def load_rate_engine():
    """Load zone lookup and all carrier rate cards into memory."""
    global ZONE_DATA, RATES_DATA

    # ---- Load zone lookup ----
    if not os.path.exists(ZONE_FILE):
        raise FileNotFoundError(f"Zone lookup file not found: {ZONE_FILE}")

    zone_df = pd.read_csv(ZONE_FILE)
    zone_df[ZONE_PREFIX_COL] = zone_df[ZONE_PREFIX_COL].astype(int)
    zone_df = zone_df.set_index(ZONE_PREFIX_COL)
    ZONE_DATA = zone_df.to_dict(orient="index")

    # ---- Load carrier rate cards ----
    for carrier, path in RATE_FILES.items():
        if not os.path.exists(path):
            print(f"[rate_engine] Missing rate file: {carrier}")
            continue

        df = pd.read_csv(path)
        df = df.sort_values(by=RATE_WEIGHT_COL).reset_index(drop=True)
        RATES_DATA[carrier] = df

    print(f"[rate_engine] Loaded {len(RATES_DATA)} carrier rate cards")


# ---------------- HELPERS ---------------- #

def zipcode_to_prefix(zipcode: int) -> int:
    """ZIP5 → ZIP3"""
    return int(str(zipcode)[:3])


def get_price(df: pd.DataFrame, weight: float, zone: int) -> float | None:
    """Return carrier price for given weight + zone."""
    col = str(zone)
    if col not in df.columns:
        return None

    effective_weight = int(np.ceil(weight))
    idx = df[RATE_WEIGHT_COL].searchsorted(effective_weight, side="left")

    if idx >= len(df):
        return None

    price = df.loc[idx, col]
    return None if pd.isna(price) else float(price)


# ---------------- CORE ENGINE ---------------- #

def find_best_rate(zipcode: int, weight: float) -> Dict:
    """
    Deterministic rate selection:
    - Evaluate ALL hubs independently
    - Select cheapest carrier per hub
    - Select cheapest hub overall
    """

    zip3 = zipcode_to_prefix(zipcode)
    zone_map = ZONE_DATA.get(zip3)

    if not zone_map:
        return {"error": "Invalid ZIP code or no zone mapping found"}

    effective_weight = int(np.ceil(weight))

    hub_results: List[Dict] = []
    debug_rows: List[Dict] = []

    # ---- Evaluate each hub independently ----
    for hub, zone in zone_map.items():
        rows = []

        for carrier, df in RATES_DATA.items():
            price = get_price(df, weight, zone)
            if price is None:
                continue

            row = {
                "carrier": carrier,
                "hub": hub,
                "zone": zone,
                "effective_weight_lb": effective_weight,
                "price": round(price, 2),
            }

            rows.append(row)
            debug_rows.append(row)

        if not rows:
            continue

        df_cmp = pd.DataFrame(rows)
        best = df_cmp.loc[df_cmp["price"].idxmin()]

        hub_results.append({
            "hub": hub,
            "zone": zone,
            "best_carrier": best["carrier"],
            "best_price": float(best["price"]),
            "comparison_table": df_cmp.sort_values("price").to_dict(orient="records")
        })

    if not hub_results:
        return {"error": "No rates found across any hub"}

    # ---- Final selection across hubs ----
    best_overall = min(hub_results, key=lambda x: x["best_price"])

    # ---- DEBUG CSV (for Excel verification) ----
    debug_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(debug_dir, exist_ok=True)
    debug_path = os.path.join(debug_dir, "rate_comparison.csv")

    pd.DataFrame(debug_rows).sort_values(
        ["hub", "price"]
    ).to_csv(debug_path, index=False)

    print(f"[rate_engine] Debug CSV written to {debug_path}")

    # ---- Final structured output ----
    return {
        "input_zipcode": zipcode,
        "zip3": zip3,
        "input_weight_lb": round(weight, 2),
        "effective_weight_lb": effective_weight,

        # Excel-equivalent breakdown
        "per_hub_results": hub_results,

        # Final answer
        "best_hub": best_overall["hub"],
        "best_zone": best_overall["zone"],
        "best_carrier": best_overall["best_carrier"],
        "best_price": round(best_overall["best_price"], 2),

        # Explain-mode metadata (LLM reads only)
        "calculation_logic": {
            "effective_weight": "ceil(actual_weight_lb)",
            "zip3": "first 3 digits of ZIP5",
            "carrier_rule": "minimum price per hub",
            "hub_rule": "minimum price across hubs"
        }
    }


# ---------------- INIT ---------------- #

load_rate_engine()
