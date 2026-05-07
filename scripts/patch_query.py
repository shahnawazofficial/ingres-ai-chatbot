"""
One-time patch: replace _query_df in api/main.py with the fixed version
that scans the query string for known state/district names (bidirectional).
"""
import os, re

MAIN_PY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "api", "main.py")

NEW_FUNC = '''def _query_df(df: "pd.DataFrame", query: str) -> "pd.DataFrame":
    """
    Fast pandas query routing - returns the most relevant rows for a query.
    Scans the query for any known state/district names mentioned within it
    (bidirectional matching - fixes sentence-length queries like
    "Compare Bihar and Uttar Pradesh groundwater status").
    """
    q = query.lower().strip()

    # -- State abbreviation expansion ----------------------------------------
    STATE_ABBR = {
        " up ": "uttar pradesh", "^up$": "uttar pradesh", "up groundwater": "uttar pradesh",
        " mp ": "madhya pradesh", "^mp$": "madhya pradesh",
        " wb ": "west bengal",   "^wb$": "west bengal",
        " ap ": "andhra pradesh","^ap$": "andhra pradesh",
        " hp ": "himachal pradesh",
        " jk ": "jammu kashmir",
    }
    import re as _re
    for abbr, full in STATE_ABBR.items():
        pattern = abbr if abbr.startswith("^") else abbr.strip()
        if pattern in q or _re.search(abbr, q):
            q = q.replace(pattern.strip(), full)

    # 1. Scan query for any known state names mentioned WITHIN it.
    #    e.g. "compare Bihar and Uttar Pradesh" -> ["bihar", "uttar pradesh"]
    #    FIX: old code did str.contains(full_query) on state column - always 0
    #         for sentence queries. Now we check query.contains(state_name).
    all_states_lower = df["state"].str.lower().unique()
    mentioned_states = [s for s in all_states_lower if s in q]
    if mentioned_states:
        state_mask = df["state"].str.lower().isin(mentioned_states)
        return df[state_mask]   # ALL districts of every mentioned state

    # 2. Short query only - also check if the query appears INSIDE a state name
    #    (e.g. query="andhra" matches "Andhra Pradesh")
    if len(q.split()) <= 3:
        state_mask = df["state"].str.lower().str.contains(q, na=False, regex=False)
        if state_mask.any():
            return df[state_mask]

    # 3. Scan query for any known district names mentioned within it
    all_dists_lower = df["district"].str.lower().unique()
    mentioned_dists = [d for d in all_dists_lower if d in q]
    if mentioned_dists:
        dist_mask = df["district"].str.lower().isin(mentioned_dists)
        return df[dist_mask]

    # 4. Short query - also check if query appears inside a district name
    if len(q.split()) <= 3:
        dist_mask = df["district"].str.lower().str.contains(q, na=False, regex=False)
        if dist_mask.any():
            return df[dist_mask].head(12)

    # 5. Category keyword routing
    if any(w in q for w in ("over-exploit", "overexploit", "critical", "urgent", "stress", "worst", "highest", "dangerous")):
        return df.nlargest(20, "stage_pct")[["state", "district", "stage_pct", "category"]]
    if any(w in q for w in ("safe", "good", "best", "lowest", "healthy")):
        return df.nsmallest(20, "stage_pct")[["state", "district", "stage_pct", "category"]]
    if any(w in q for w in ("semi", "moderate")):
        return df[df["category"] == "Semi-Critical"].head(20)
    if any(w in q for w in ("all state", "every state", "list state", "how many state")):
        return df.sort_values("stage_pct", ascending=False).drop_duplicates("state")[["state", "district", "stage_pct", "category"]]

    # 6. Fallback - top 25 most stressed districts across all states
    return df.nlargest(25, "stage_pct")[["state", "district", "stage_pct", "category"]]
'''

src = open(MAIN_PY, "r", encoding="utf-8").read()

# Find the exact start and end of the old _query_df function
start_marker = 'def _query_df('
end_marker   = '\n# -- Cache'   # next section starts here

start_idx = src.find(start_marker)
end_idx   = src.find('\n# \u2500\u2500 Cache')  # the Unicode box-drawing separator

if start_idx == -1:
    print("ERROR: Could not find def _query_df in file")
    raise SystemExit(1)
if end_idx == -1:
    # Try ASCII fallback
    end_idx = src.find('\ndef _cache_key')
    if end_idx == -1:
        print("ERROR: Could not find end boundary")
        raise SystemExit(1)

old_func = src[start_idx:end_idx]
print(f"Found _query_df: chars {start_idx}-{end_idx} ({len(old_func)} chars)")
print(f"Old function first line: {old_func.splitlines()[0]}")

new_src = src[:start_idx] + NEW_FUNC + "\n" + src[end_idx:]
open(MAIN_PY, "w", encoding="utf-8").write(new_src)
print("SUCCESS: api/main.py patched with fixed _query_df")

# Quick sanity check
import importlib.util
spec = importlib.util.spec_from_file_location("main_check", MAIN_PY)
# just parse it
import ast
try:
    ast.parse(open(MAIN_PY, encoding="utf-8").read())
    print("SYNTAX OK: api/main.py parses without errors")
except SyntaxError as e:
    print(f"SYNTAX ERROR: {e}")
