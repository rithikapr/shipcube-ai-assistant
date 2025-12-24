# utils/rate_engine.py

import pandas as pd
import numpy as np
import os
from typing import Dict

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

# ---------------- LOAD DATA (ONCE) ---------------- #

def load_rate_engine():
    """Load zone lookup and all carrier rate cards into memory."""
    global ZONE_DATA, RATES_DATA

    # ---- Load zone lookup (ZIP3 based) ----
    if not os.path.exists(ZONE_FILE):
        raise FileNotFoundError(f"Zone lookup file not found: {ZONE_FILE}")

    zone_df = pd.read_csv(ZONE_FILE)
    zone_df[ZONE_PREFIX_COL] = zone_df[ZONE_PREFIX_COL].astype(int)
    zone_df = zone_df.set_index(ZONE_PREFIX_COL)

    ZONE_DATA = zone_df.to_dict(orient="index")

    # ---- Load carrier rate cards ----
    for carrier, path in RATE_FILES.items():
        if not os.path.exists(path):
            print(f"[rate_engine] Missing rate file for {carrier}: {path}")
            continue

        df = pd.read_csv(path)
        df = df.sort_values(by=RATE_WEIGHT_COL).reset_index(drop=True)
        RATES_DATA[carrier] = df

    print(f"[rate_engine] Loaded {len(RATES_DATA)} carriers")


# ---------------- HELPERS ---------------- #

def zipcode_to_prefix(zipcode: int) -> int:
    """Convert ZIP5 → ZIP3"""
    return int(str(zipcode)[:3])


def get_price(df: pd.DataFrame, weight: float, zone: int) -> float | None:
    """Get price for weight and zone from carrier rate card."""
    col = str(zone)
    if col not in df.columns:
        return None

    effective_weight = int(np.ceil(weight))
    idx = df[RATE_WEIGHT_COL].searchsorted(effective_weight, side="left")

    if idx >= len(df):
        return None

    price = df.loc[idx, col]
    if pd.isna(price):
        return None

    return float(price)


# ---------------- CORE API ---------------- #

def find_best_rate(zipcode: int, weight: float) -> Dict:
    zip3 = zipcode_to_prefix(zipcode)
    zone_map = ZONE_DATA.get(zip3)

    if not zone_map:
        return {"error": "Invalid zipcode or zone not found"}

    hub = min(zone_map, key=zone_map.get)
    zone = zone_map[hub]

    effective_weight = int(np.ceil(weight))
    comparisons = []

    for carrier, df in RATES_DATA.items():
        price = get_price(df, weight, zone)
        if price is not None:
            comparisons.append({
                "carrier": carrier,
                "zone": zone,
                "effective_weight": effective_weight,
                "price": round(float(price), 2)
            })

    if not comparisons:
        return {"error": "No rates found"}

    df_cmp = pd.DataFrame(comparisons)
    best = df_cmp.loc[df_cmp["price"].idxmin()]

    return {
        "zipcode": zipcode,
        "zip3": zip3,
        "input_weight_lb": round(weight, 2),
        "effective_weight_lb": effective_weight,
        "hub": hub,
        "zone": zone,
        "best_carrier": best["carrier"],
        "best_price": best["price"],
        "comparison_table": df_cmp.sort_values("price").to_dict(orient="records"),
        "calculation_logic": {
            "effective_weight_formula": "ceil(actual_weight_lb)",
            "zip3_formula": "first 3 digits of ZIP5",
            "carrier_selection_rule": "minimum price across eligible carriers"
        }
    }

    # ---- Debug CSV (optional but useful) ----
    debug_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(debug_dir, exist_ok=True)
    debug_path = os.path.join(debug_dir, "rate_comparison.csv")
    df_cmp.to_csv(debug_path, index=False)

    print(f"[rate_engine] ZIP5={zipcode} ZIP3={zip3} Zone={zone}")
    print(f"[rate_engine] Debug CSV written to: {debug_path}")

    return {
        "zipcode": zipcode,
        "weight": weight,
        "best_carrier": best["carrier"],
        "best_price": round(best["price"], 2),
        "hub": best["hub"],
        "zone": int(best["zone"]),
        "all_options": df_cmp.sort_values("price").to_dict(orient="records")
    }


# ---------------- INIT ON IMPORT ---------------- #

load_rate_engine()
