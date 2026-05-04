#!/usr/bin/env python3
"""
scripts/rebuild_chromadb.py
===========================
Rebuilds the ChromaDB vector store from the clean 634-district CSV.
Each district becomes a rich text document with all key metrics.

Run: python scripts/rebuild_chromadb.py
"""
import os, sys
import pandas as pd
import chromadb

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "data", "processed", "groundwater_clean.csv")
EMB_PATH = os.path.join(ROOT, "data", "embeddings")

def make_doc(row) -> str:
    """Convert a district row into a rich searchable text document."""
    avail = f"{row['net_gw_availability_ham']:.1f}" if pd.notna(row.get('net_gw_availability_ham')) else "N/A"
    irrig = f"{row['net_gw_irrigation_ham']:.1f}" if pd.notna(row.get('net_gw_irrigation_ham')) else "N/A"
    stage = f"{row['stage_pct']:.2f}" if pd.notna(row.get('stage_pct')) else "N/A"
    cat   = row.get('category', 'Unknown')
    state = row['state']
    dist  = row['district']

    return (
        f"State: {state}\n"
        f"District: {dist}\n"
        f"Net Groundwater Availability: {avail} ham\n"
        f"Net GW Available for Irrigation: {irrig} ham\n"
        f"Stage of GW Development: {stage}%\n"
        f"Category: {cat}\n"
        f"The district {dist} in {state} has a groundwater development stage of {stage}% "
        f"and is classified as '{cat}'. Net groundwater availability is {avail} ham."
    )

def main():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: CSV not found at {CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    df["state"]    = df["state"].str.strip().str.title()
    df["district"] = df["district"].str.strip().str.title()
    print(f"Loaded {len(df)} districts from CSV.")

    # Connect to ChromaDB and reset the collection
    client = chromadb.PersistentClient(path=EMB_PATH)

    # Delete old collection if exists
    try:
        client.delete_collection("groundwater_data")
        print("Deleted old collection.")
    except Exception:
        pass

    col = client.create_collection(
        name="groundwater_data",
        metadata={"hnsw:space": "cosine"},
    )
    print("Created fresh collection.")

    # Build documents in batches of 100
    BATCH = 100
    total = 0
    for start in range(0, len(df), BATCH):
        batch = df.iloc[start : start + BATCH]
        docs, ids, metas = [], [], []

        for _, row in batch.iterrows():
            doc_id = f"{row['state'].lower().replace(' ','_')}_{row['district'].lower().replace(' ','_')}"
            docs.append(make_doc(row))
            ids.append(doc_id)
            metas.append({
                "state":    row["state"],
                "district": row["district"],
                "stage":    float(row["stage_pct"]) if pd.notna(row.get("stage_pct")) else 0.0,
                "category": row.get("category", ""),
                "source":   f"{row['state']} - {row['district']}",
            })

        col.add(documents=docs, ids=ids, metadatas=metas)
        total += len(batch)
        print(f"  Indexed {total}/{len(df)} districts...")

    print(f"\nDone! ChromaDB now has {col.count()} district documents.")
    print(f"Path: {EMB_PATH}")

if __name__ == "__main__":
    main()
