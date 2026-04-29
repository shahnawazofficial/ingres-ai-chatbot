#!/usr/bin/env python3
"""
scripts/preprocess.py
=====================
One-time preprocessing pipeline for INGRES groundwater data.
Run this once to produce data/processed/groundwater_clean.csv

Usage:
    python scripts/preprocess.py

Output:
    data/processed/groundwater_clean.csv  ← fast, clean, production-ready CSV
"""

import os
import sys
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV  = os.path.join(ROOT, "data", "processed", "groundwater_data.csv")
OUT_CSV  = os.path.join(ROOT, "data", "processed", "groundwater_clean.csv")

# ── Helpers ────────────────────────────────────────────────────────────────
def _safe_float(v):
    try:    return float(str(v).strip())
    except: return None

def _categorize(stage: float) -> str:
    if stage < 70:    return "Safe"
    elif stage < 90:  return "Semi-Critical"
    elif stage <= 100: return "Critical"
    else:             return "Over-Exploited"

# ── Parser ─────────────────────────────────────────────────────────────────
def parse(path: str) -> pd.DataFrame:
    print(f"Reading raw CSV: {path}")
    raw = pd.read_csv(path, header=0, low_memory=False)
    print(f"  Raw shape: {raw.shape}")

    # Step 1 — state list from summary rows 4-42
    states_ordered = []
    for i in range(4, 43):
        row = raw.iloc[i]
        c1 = str(row.iloc[1]).strip()
        skip = {"nan","none","total states","union territories","total uts","grand total",""}
        if c1.lower() not in skip and any(ch.isalpha() for ch in c1):
            states_ordered.append(c1)
    print(f"  States found: {len(states_ordered)}")

    # Step 2 — find section delimiters (col15 = 'State total (ham)')
    total_rows = [
        i for i in range(43, len(raw))
        if "state total" in str(raw.iloc[i, 15]).lower()
        and "(ham)" in str(raw.iloc[i, 15]).lower()
    ]
    print(f"  Section delimiters: {len(total_rows)}")

    # Step 3 — extract district records per block
    records = []
    block_starts = [43] + [e + 2 for e in total_rows[:-1]]
    block_ends   = total_rows

    for state_idx, (blk_s, blk_e) in enumerate(zip(block_starts, block_ends)):
        if state_idx >= len(states_ordered):
            break
        state_name = states_ordered[state_idx]

        for i in range(blk_s, blk_e):
            row  = raw.iloc[i]
            dist = str(row.iloc[16]).strip()
            stag = str(row.iloc[21]).strip()

            if not dist or dist.lower() in ("nan","none",""):
                continue
            if not any(ch.isalpha() for ch in dist):
                continue
            if any(kw in dist.lower() for kw in ("total","district","sl.","parameter")):
                continue

            stage = _safe_float(stag)
            if stage is None or not (0.0 <= stage <= 500.0):
                continue

            records.append({
                "state":                   state_name.strip().title(),
                "district":                dist.strip().title(),
                "net_gw_availability_ham": _safe_float(str(row.iloc[18]).strip()),
                "net_gw_irrigation_ham":   _safe_float(str(row.iloc[20]).strip()),
                "stage_pct":               round(stage, 2),
                "category":                _categorize(stage),
            })

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["state", "district"])
    print(f"  Districts extracted: {len(df)}")
    return df

# ── Validation ─────────────────────────────────────────────────────────────
def validate(df: pd.DataFrame) -> bool:
    ok = True
    if df.empty:
        print("ERROR: DataFrame is empty"); return False
    if len(df) < 100:
        print(f"WARN: Only {len(df)} rows — expected 600+")
    missing_state = df["state"].isna().sum()
    missing_dist  = df["district"].isna().sum()
    if missing_state > 0:
        print(f"WARN: {missing_state} rows with missing state")
    if missing_dist > 0:
        print(f"WARN: {missing_dist} rows with missing district")
    stage_invalid = ((df["stage_pct"] < 0) | (df["stage_pct"] > 500)).sum()
    if stage_invalid > 0:
        print(f"ERROR: {stage_invalid} rows with invalid stage_pct"); ok = False

    print(f"\n  Category breakdown:")
    for cat, cnt in df["category"].value_counts().items():
        print(f"    {cat}: {cnt} ({cnt/len(df)*100:.1f}%)")
    return ok

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(RAW_CSV):
        print(f"ERROR: Source CSV not found: {RAW_CSV}")
        sys.exit(1)

    df = parse(RAW_CSV)
    print("\nValidation:")
    if not validate(df):
        print("Validation failed — not saving.")
        sys.exit(1)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved clean CSV → {OUT_CSV}")
    print(f"Rows: {len(df)} | Columns: {list(df.columns)}")

if __name__ == "__main__":
    main()
