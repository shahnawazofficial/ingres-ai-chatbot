"""
INGRES Debug Script — verify CSV and ChromaDB state locally.
Run: python scripts/debug_data.py
"""
import os, sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(ROOT, "data", "processed")
EMB_PATH  = os.path.join(ROOT, "data", "embeddings")

sep = "=" * 60

# ── 1. CSV Audit ────────────────────────────────────────────────────────────
print(sep)
print("1. CSV FILES")
print(sep)

csv_files = {
    "groundwater_clean.csv": os.path.join(PROCESSED, "groundwater_clean.csv"),
    "groundwater_full.csv":  os.path.join(PROCESSED, "groundwater_full.csv"),
    "groundwater_data.csv":  os.path.join(PROCESSED, "groundwater_data.csv"),
}

for fname, fpath in csv_files.items():
    if not os.path.exists(fpath):
        print(f"  [MISSING] {fname}")
        continue
    size_kb = os.path.getsize(fpath) / 1024
    df = pd.read_csv(fpath)
    print(f"\n  [{fname}]  ({size_kb:.1f} KB)")
    print(f"    Rows    : {len(df)}")
    print(f"    Columns : {list(df.columns)}")
    if "state" in df.columns:
        df["state"] = df["state"].str.strip().str.title()
        n_states = df["state"].nunique()
        print(f"    States  : {n_states}")
        up_mask = df["state"].str.lower().str.contains("uttar pradesh", na=False)
        print(f"    UP rows : {up_mask.sum()}")
        if up_mask.sum() > 0 and "district" in df.columns:
            districts = df.loc[up_mask, "district"].str.strip().str.title().tolist()
            print(f"    UP districts: {districts[:10]} ...")
        # Print all unique states
        states = sorted(df["state"].unique())
        print(f"    All states: {states}")

# ── 2. ChromaDB Audit ───────────────────────────────────────────────────────
print()
print(sep)
print("2. CHROMADB")
print(sep)

sqlite_path = os.path.join(EMB_PATH, "chroma.sqlite3")
print(f"  Embeddings path : {EMB_PATH}")
print(f"  sqlite3 exists  : {os.path.exists(sqlite_path)}")
if os.path.exists(sqlite_path):
    print(f"  sqlite3 size    : {os.path.getsize(sqlite_path)/1024:.1f} KB")

try:
    import chromadb
    client = chromadb.PersistentClient(path=EMB_PATH)
    collections = client.list_collections()
    print(f"  Collections     : {[c.name for c in collections]}")
    for col in collections:
        c = client.get_collection(col.name)
        count = c.count()
        print(f"\n  Collection '{col.name}':")
        print(f"    Total docs  : {count}")
        if count > 0:
            # Sample a few records
            sample = c.get(limit=5, include=["metadatas"])
            metas = sample.get("metadatas", [])
            print(f"    Sample meta : {metas[:3]}")
            # Check if Uttar Pradesh exists
            try:
                up_results = c.get(
                    where={"state": "Uttar Pradesh"},
                    include=["metadatas"]
                )
                up_count = len(up_results.get("ids", []))
                print(f"    UP docs     : {up_count}")
            except Exception as e:
                print(f"    UP query err: {e}")
                # Try a query search instead
                try:
                    qr = c.query(
                        query_texts=["uttar pradesh groundwater"],
                        n_results=5,
                        include=["metadatas", "documents"]
                    )
                    docs  = qr.get("documents", [[]])[0]
                    metas2 = qr.get("metadatas", [[]])[0]
                    print(f"    UP query results ({len(docs)}): {[m.get('state','?') for m in metas2]}")
                except Exception as e2:
                    print(f"    UP query fallback err: {e2}")

            # List distinct states in ChromaDB
            try:
                all_meta = c.get(include=["metadatas"])
                all_metas_list = all_meta.get("metadatas", [])
                chroma_states = set(m.get("state", "?") for m in all_metas_list if isinstance(m, dict))
                print(f"    States in DB: {sorted(chroma_states)}")
            except Exception as e:
                print(f"    States list err: {e}")

except ImportError:
    print("  chromadb NOT installed")
except Exception as e:
    print(f"  ChromaDB error: {e}")

# ── 3. Git Status ───────────────────────────────────────────────────────────
print()
print(sep)
print("3. GIT TRACKED FILES (data/)")
print(sep)
os.system(f"cd /d {ROOT} && git ls-files data/ --error-unmatch 2>&1 | head -40 || git ls-files data/")

print()
print(sep)
print("DONE")
print(sep)
