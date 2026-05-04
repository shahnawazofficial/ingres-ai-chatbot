#!/usr/bin/env python3
"""
scripts/extract_groundwater.py
==============================
Robust PDF extraction pipeline for district-wise groundwater data.

Strategy
--------
* Annexure-II district tables span pages 4–41 (indices 3–40).
* Each table page repeats a 4-row header block (col headers + col numbers).
* The page's plain text contains the STATE NAME in the first ~3 lines
  (e.g. "GROUND WATER RESOURCES … \nANDHRA PRADESH\n").
* Data rows have 15 columns: Sl.No, District, cols 3-15.
* "State Total" rows mark end-of-state; they are skipped.
* Column layout (1-indexed as in PDF):
    1  Sl.No
    2  District
    3  Monsoon Recharge from Rainfall      (ham)
    4  Monsoon Recharge from Other Sources (ham)
    5  Non-Monsoon Recharge from Rainfall  (ham)
    6  Non-Monsoon Recharge from Other Sources (ham)
    7  Total Recharge                       (ham)
    8  Natural Discharge during Non-Monsoon (ham)
    9  Net GW Availability                  (ham)
   10  Annual GW Draft – Irrigation         (ham)
   11  Annual GW Draft – Domestic & Industrial (ham)
   12  Annual GW Draft – Total              (ham)
   13  Projected Demand (Domestic+Ind) upto 2025 (ham)
   14  Net GW Availability for Future Irrigation (ham)
   15  Stage of GW Development (%)

Usage
-----
    python scripts/extract_groundwater.py

Outputs
-------
    data/processed/groundwater_full.csv
    data/processed/groundwater_full.xlsx
    data/processed/groundwater_clean.csv   (replaces old file – minimal cols)
"""

import os
import re
import sys
import unicodedata

import pandas as pd
import pdfplumber

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_PATH = os.path.join(ROOT, "data", "raw", "groundwater3_full.pdf")
OUT_DIR  = os.path.join(ROOT, "data", "processed")
FULL_CSV  = os.path.join(OUT_DIR, "groundwater_full.csv")
FULL_XLSX = os.path.join(OUT_DIR, "groundwater_full.xlsx")
CLEAN_CSV = os.path.join(OUT_DIR, "groundwater_clean.csv")

# Pages that contain Annexure-II district tables (1-indexed, inclusive)
DISTRICT_PAGE_START = 4   # page 4
DISTRICT_PAGE_END   = 41  # page 41

# ── Constants ─────────────────────────────────────────────────────────────────
SKIP_KEYWORDS = {
    "state total", "statetotal", "ut total", "grand total", "total (ham)", "total (bcm)",
    "sl.", "sl.no", "monsoon", "non monsoon", "recharge",
    "natural discharge", "irrigation use", "domestic supply", "projected demand", "annual replenishable",
    "ground water resources", "replenishable", "utilization", "stage of ground",
    "union territories",
}

# Known Indian states/UTs in rough order of appearance in the PDF
# This is used as a fallback when page-text extraction cannot find the state name.
STATE_ORDER = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Delhi", "Goa", "Gujarat", "Haryana", "Himachal Pradesh",
    "Jammu & Kashmir", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh",
    "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland",
    "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
    "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
    # Union Territories
    "Andaman & Nicobar", "Chandigarh", "Dadra & Nagar Haveli",
    "Daman & Diu", "Lakshadweep", "Puducherry",
]

# Alternate spellings / OCR variants that map to canonical names
STATE_ALIASES = {
    "orissa":           "Odisha",
    "odisha":           "Odisha",
    "uttrakhand":       "Uttarakhand",
    "uttarakhand":      "Uttarakhand",
    "madyha pradesh":   "Madhya Pradesh",
    "madhya pradesh":   "Madhya Pradesh",
    "jammu kashmir":    "Jammu & Kashmir",
    "jammu and kashmir": "Jammu & Kashmir",
    "andaman nicobar":  "Andaman & Nicobar",
    "andaman and nicobar": "Andaman & Nicobar",
    "dadra nagar haveli": "Dadra & Nagar Haveli",
    "daman diu":        "Daman & Diu",
    "goa":              "Goa",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def normalize(s: str) -> str:
    """Normalize unicode, collapse whitespace, strip."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def safe_float(v) -> float | None:
    """Convert a cell value to float, return None if not possible."""
    if v is None:
        return None
    s = normalize(str(v)).replace(",", "").replace("\u2212", "-")
    try:
        return float(s)
    except ValueError:
        return None


def categorize(stage: float) -> str:
    if stage < 70:
        return "Safe"
    elif stage < 90:
        return "Semi-Critical"
    elif stage <= 100:
        return "Critical"
    else:
        return "Over-Exploited"


def is_skip_row(row: list) -> bool:
    """True if this row is a header, total, or otherwise non-data row."""
    col1 = normalize(str(row[0] or "")).lower()
    col2 = normalize(str(row[1] or "")).lower()

    # Header row: col1 is 'sl.\nno.' or column-number indicator
    if col1 in ("sl.\nno.", "sl.no.", "1", ""):
        if col2 in ("2", "district", ""):
            return True

    # State/UT total rows
    combined = (col1 + " " + col2).lower()
    for kw in ("state total", "statetotal", "ut total", "uttotal", "grand total", "union territories"):
        if kw in combined:
            return True

    # Row where sl_no is not a digit (header continuation rows)
    if col1 and not re.match(r"^\d+$", col1):
        for kw in SKIP_KEYWORDS:
            if kw in col1:  # Only check col1 (sl.no), NOT col2 (district name)
                return True

    return False


def is_data_row(row: list) -> bool:
    """True if the row looks like a valid district data row."""
    if len(row) < 15:
        return False
    if is_skip_row(row):
        return False
    # Must have a numeric Sl.No in col[0]
    sl = normalize(str(row[0] or ""))
    if not re.match(r"^\d+$", sl):
        return False
    # Must have a district/block name (alphabetic) in col[1]
    dist = normalize(str(row[1] or ""))
    if not dist or not any(c.isalpha() for c in dist):
        return False
    # Reject only if the FIRST word of district is a known admin keyword
    # (avoid dropping "Bishnupur District*" or "Churachandpur District*")
    dist_lower = dist.lower()
    bad_starts = ("sl.", "monsoon", "non monsoon", "recharge", "natural discharge",
                  "irrigation use", "domestic", "projected", "annual replenishable",
                  "ground water", "replenishable", "utilization", "stage of")
    for kw in bad_starts:
        if dist_lower.startswith(kw):
            return False
    return True


def extract_state_from_text(text: str) -> str | None:
    """
    Extract the state/UT name from the page header text.
    The header typically looks like:
        GROUND WATER RESOURCES AVAILABILITY …
        ANDHRA PRADESH
    We look for an ALL-CAPS (or Title Case) line that is not part of
    the repeating column-header boilerplate.
    """
    BOILERPLATE = {
        "GROUND WATER RESOURCES AVAILABILITY",
        "UTILIZATION AND STAGE OF DEVELOPMENT",
        "UTILIZATION AND STAGE OF DEVELOPMET",
        "GROUND WATER RESOURCES",
        "AVAILABILITY",
        "STAGE OF DEVELOPMENT",
        "ANNEXURE",
        "DISTRICT-WISE",
    }
    lines = [normalize(ln) for ln in (text or "").splitlines() if normalize(ln)]
    for line in lines[:10]:  # State name is always in top lines
        upper = line.upper()
        # Skip if it's a boilerplate line
        skip = False
        for bp in BOILERPLATE:
            if bp in upper:
                skip = True
                break
        if skip:
            continue
        # Skip lines with digits (col numbers, page numbers)
        if any(c.isdigit() for c in line):
            continue
        # Must contain only alpha/space/&/(/)
        if re.match(r"^[A-Za-z\s&()/'-]+$", line) and len(line) >= 3:
            return line.title()
    return None


# ── Core extraction ───────────────────────────────────────────────────────────
def extract_all_pages(pdf_path: str) -> pd.DataFrame:
    records = []
    current_state = None
    state_fallback_idx = 0

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"PDF has {total_pages} pages. Processing pages {DISTRICT_PAGE_START}–{DISTRICT_PAGE_END}.")

        for page_num in range(DISTRICT_PAGE_START, DISTRICT_PAGE_END + 1):
            page_idx = page_num - 1  # 0-indexed
            if page_idx >= total_pages:
                break

            page = pdf.pages[page_idx]

            # ── Detect state from page header text ──────────────────────────
            page_text = page.extract_text() or ""
            detected_state = extract_state_from_text(page_text)

            if detected_state:
                # Confirm it actually looks like a known state (fuzzy match)
                matched = _match_known_state(detected_state)
                if matched and matched != current_state:
                    print(f"  Page {page_num:3d}: state -> {matched}")
                    current_state = matched
                elif not matched and detected_state != current_state:
                    # Use as-is if no match (rare UTs)
                    print(f"  Page {page_num:3d}: state (unmatched) -> {detected_state}")
                    current_state = detected_state

            if current_state is None:
                print(f"  Page {page_num:3d}: WARNING – no state detected, skipping page.")
                continue

            # ── Extract table ────────────────────────────────────────────────
            rows = page.extract_table() or []
            page_records = 0

            for row in rows:
                if not is_data_row(row):
                    continue

                district = normalize(str(row[1])).strip("*").strip()

                rec = {
                    "state":                    current_state,
                    "district":                 district.title(),
                    "monsoon_recharge_rainfall": safe_float(row[2]),
                    "monsoon_recharge_other":    safe_float(row[3]),
                    "nonmonsoon_recharge_rainfall": safe_float(row[4]),
                    "nonmonsoon_recharge_other": safe_float(row[5]),
                    "total_recharge_ham":        safe_float(row[6]),
                    "natural_discharge_ham":     safe_float(row[7]),
                    "net_gw_availability_ham":   safe_float(row[8]),
                    "gw_draft_irrigation_ham":   safe_float(row[9]),
                    "gw_draft_domestic_ind_ham": safe_float(row[10]),
                    "gw_draft_total_ham":        safe_float(row[11]),
                    "projected_demand_2025_ham": safe_float(row[12]),
                    "net_gw_future_irrigation_ham": safe_float(row[13]),
                    "stage_pct":                 safe_float(row[14]),
                }

                # Validate: must have a numeric stage
                if rec["stage_pct"] is None:
                    continue
                # Accept any positive stage value (Kala Amb Valley = 565%, some OE > 500%)
                if not (0.0 <= rec["stage_pct"] <= 1000.0):
                    continue

                rec["category"] = categorize(rec["stage_pct"])
                records.append(rec)
                page_records += 1

            print(f"  Page {page_num:3d}: {page_records} district rows extracted.")

    df = pd.DataFrame(records)
    return df


def _match_known_state(name: str) -> str | None:
    """Fuzzy-match a detected state name to our known list."""
    name_clean = re.sub(r"[^a-z ]", "", name.lower()).strip()

    # 1. Check alias table first (handles OCR typos and alternate spellings)
    for alias, canonical in STATE_ALIASES.items():
        alias_clean = re.sub(r"[^a-z ]", "", alias.lower()).strip()
        if name_clean == alias_clean or alias_clean in name_clean or name_clean in alias_clean:
            return canonical

    name_upper = name.upper().strip()
    for known in STATE_ORDER:
        known_upper = known.upper()
        # Exact
        if name_upper == known_upper:
            return known
        # One contains the other
        if known_upper in name_upper or name_upper in known_upper:
            return known
        # Normalize & compare without spaces / special chars
        n1 = re.sub(r"[^A-Z]", "", name_upper)
        n2 = re.sub(r"[^A-Z]", "", known_upper)
        if n1 == n2:
            return known
    return None


# ── Deduplication & validation ────────────────────────────────────────────────
def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\nRaw extracted rows: {len(df)}")

    # Remove exact duplicates
    df = df.drop_duplicates()

    # Remove (state, district) duplicates, keeping first occurrence
    before = len(df)
    df = df.drop_duplicates(subset=["state", "district"])
    after = len(df)
    if before != after:
        print(f"  Dropped {before - after} duplicate (state, district) entries.")

    print(f"Final unique districts: {len(df)}")
    print(f"\nState breakdown:")
    state_counts = df.groupby("state").size().sort_index()
    for state, cnt in state_counts.items():
        print(f"  {state:<30s} {cnt:>4d} districts")

    print(f"\nCategory breakdown:")
    for cat, cnt in df["category"].value_counts().items():
        print(f"  {cat:<20s} {cnt:>4d} ({cnt/len(df)*100:.1f}%)")

    # Validate numeric columns
    issues = []
    numeric_cols = [c for c in df.columns if c not in ("state", "district", "category")]
    for col in numeric_cols:
        null_count = df[col].isna().sum()
        if null_count > 0:
            issues.append(f"  Column '{col}' has {null_count} null values.")

    if issues:
        print("\nData quality notes:")
        for issue in issues:
            print(issue)

    return df


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(PDF_PATH):
        print(f"ERROR: PDF not found at {PDF_PATH}")
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("INGRES Groundwater PDF Extraction Pipeline")
    print("=" * 60)

    df = extract_all_pages(PDF_PATH)

    if df.empty:
        print("ERROR: No records extracted! Check PDF structure.")
        sys.exit(1)

    df = validate_and_clean(df)

    # ── Save full dataset ────────────────────────────────────────────────────
    df.to_csv(FULL_CSV, index=False)
    print(f"\nFull dataset saved → {FULL_CSV}  ({len(df)} rows, {len(df.columns)} cols)")

    try:
        df.to_excel(FULL_XLSX, index=False, engine="openpyxl")
        print(f"Full dataset saved -> {FULL_XLSX}")
    except ImportError:
        print("NOTE: openpyxl not installed; skipping .xlsx export. Install with: pip install openpyxl")

    # ── Save slim clean CSV (backward-compatible with existing pipeline) ──────
    slim_cols = [
        "state", "district",
        "net_gw_availability_ham", "gw_draft_irrigation_ham",
        "stage_pct", "category",
    ]
    slim = df[slim_cols].rename(columns={
        "gw_draft_irrigation_ham": "net_gw_irrigation_ham",
    })
    slim.to_csv(CLEAN_CSV, index=False)
    print(f"Clean CSV saved      -> {CLEAN_CSV}  ({len(slim)} rows)")

    print("\nExtraction complete. ✔")
    print(f"Total districts: {len(df)}")
    print(f"Total states:    {df['state'].nunique()}")


if __name__ == "__main__":
    main()
