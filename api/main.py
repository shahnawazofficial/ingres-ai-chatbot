# -*- coding: utf-8 -*-
"""
FastAPI Server for INGRES Chatbot - Production v2.0
Features:
  - lifespan startup (no deprecated on_event)
  - Conversation history per session
  - /api/stats  - real CSV analytics
  - /api/search - structured data filter
  - LRU response cache
  - Enhanced RAG prompt
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional
import sys, os, requests, time, hashlib
from collections import OrderedDict
from datetime import datetime

# Ensure project root is on the path (works locally and on Railway)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dotenv import load_dotenv
load_dotenv()  # no-op on Railway (uses env vars from dashboard)

# --- Optional deps -----------------------------------------------------------
try:
    import pandas as pd
    PANDAS_OK = True
except ImportError:
    PANDAS_OK = False

try:
    import chromadb
    CHROMA_OK = True
except Exception:
    CHROMA_OK = False

# --- Global state ------------------------------------------------------------
collection = None
api_key: Optional[str] = None
df_groundwater = None
_CACHE_MAX = 200
_response_cache: OrderedDict = OrderedDict()
_conversations: dict = {}

# --- Helpers -----------------------------------------------------------------
def _safe_float(val):
    try:
        return float(val)
    except Exception:
        return None

def _categorize(stage: float) -> str:
    if stage < 70:   return "Safe"
    elif stage < 90: return "Semi-Critical"
    elif stage <= 100: return "Critical"
    else:            return "Over-Exploited"

def _load_clean_dataframe(path: str):
    """
    Parse the INGRES groundwater CSV.
    Layout discovered by inspection:
      - Rows 4-32  : State summary (col1=state name, col13=net avail bcm, col13=stage%)
      - Row 46+    : District detail blocks
          col15 = Sl.No. or 'State total (ham)' / 'StateTotal (bcm)'
          col16 = District name
          col17 = Natural discharge (ham)
          col18 = Net GW Availability (ham)
          col19 = Projected demand 2025 (ham)
          col20 = Net GW for irrigation (ham)
          col21 = Stage of GW Development (%)
    Each state block ends when col15 contains 'State total'.
    """
    raw = pd.read_csv(path, header=0, low_memory=False)

    # --- Step 1: Build ordered state list from summary rows 4-42 ---
    states_ordered = []
    for i in range(4, 43):
        row = raw.iloc[i]
        c1 = str(row.iloc[1]).strip()
        skip = {"nan", "none", "total states", "union territories",
                "total uts", "grand total", ""}
        if c1.lower() not in skip and any(ch.isalpha() for ch in c1):
            states_ordered.append(c1)

    # --- Step 2: Find 'State total' row indices (section delimiters) ---
    total_rows = []
    for i in range(43, len(raw)):
        c15 = str(raw.iloc[i, 15]).strip().lower()
        if "state total" in c15 or "statetotal" in c15:
            if "(ham)" in c15:          # only the ham row, not bcm duplicate
                total_rows.append(i)

    # --- Step 3: Extract district records per state block ---
    records = []
    state_idx = 0

    # Build block boundaries: each block ends at a 'state total' row
    # First block starts at row 46 (after 4 header rows per state)
    block_end = total_rows
    block_start = [43] + [e + 2 for e in total_rows[:-1]]   # skip bcm row after ham row

    for blk_s, blk_e in zip(block_start, block_end):
        if state_idx >= len(states_ordered):
            break
        state_name = states_ordered[state_idx]
        state_idx += 1

        for i in range(blk_s, blk_e):
            row = raw.iloc[i]
            c16 = str(row.iloc[16]).strip()   # district name
            c21 = str(row.iloc[21]).strip()   # stage %

            # Skip non-district rows
            if not c16 or c16.lower() in ("nan", "none", ""):
                continue
            if not any(ch.isalpha() for ch in c16):
                continue
            # Skip sub-header labels
            if any(kw in c16.lower() for kw in ("total", "district", "sl.", "parameter")):
                continue

            try:
                stage_num = float(c21)
                if not (0.0 <= stage_num <= 500.0):
                    continue
                records.append({
                    "state":                  state_name,
                    "district":               c16,
                    "net_gw_availability_ham":_safe_float(str(row.iloc[18]).strip()),
                    "net_gw_irrigation_ham":  _safe_float(str(row.iloc[20]).strip()),
                    "stage_pct":              stage_num,
                    "category":               _categorize(stage_num),
                })
            except (ValueError, TypeError):
                pass

    df = pd.DataFrame(records)
    if df.empty:
        return df
    df["state"]    = df["state"].str.strip().str.title()
    df["district"] = df["district"].str.strip().str.title()
    return df.drop_duplicates(subset=["state", "district"])

def _cache_key(query: str) -> str:
    return hashlib.md5(query.lower().strip().encode()).hexdigest()

def _get_cache(key: str) -> Optional[str]:
    if key in _response_cache:
        _response_cache.move_to_end(key)
        return _response_cache[key]
    return None

def _set_cache(key: str, value: str):
    _response_cache[key] = value
    _response_cache.move_to_end(key)
    if len(_response_cache) > _CACHE_MAX:
        _response_cache.popitem(last=False)

# --- Lifespan ----------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global collection, api_key, df_groundwater
    print("\n=== INGRES Chatbot API v2 Starting ===")

    api_key = os.getenv("GEMINI_API_KEY")
    print("OK  Gemini key loaded" if api_key else "ERR GEMINI_API_KEY missing")

    if CHROMA_OK:
        try:
            client = chromadb.PersistentClient(
                path=os.path.join(_project_root, "data", "embeddings")
            )
            collection = client.get_collection("groundwater_data")
            print(f"OK  Vector store: {collection.count()} docs")
        except Exception as e:
            print(f"WARN Vector store unavailable: {e}")
    else:
        print("WARN chromadb not installed")

    if PANDAS_OK:
        try:
            csv_path = os.path.join(_project_root, "data", "processed", "groundwater_data.csv")
            if os.path.exists(csv_path):
                df_groundwater = _load_clean_dataframe(csv_path)
                print(f"OK  CSV: {len(df_groundwater)} district rows")
            else:
                print(f"WARN CSV not found at {csv_path}")
        except Exception as e:
            print(f"WARN CSV load failed: {e}")

    print("Docs: http://localhost:8000/docs")
    print("="*40)
    yield
    print("Shutting down.")

# --- App ---------------------------------------------------------------------
app = FastAPI(
    title="INGRES Chatbot API",
    description="AI chatbot for India Groundwater Resource Estimation System",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Serve frontend static files ---------------------------------------------
_web_dir = os.path.join(_project_root, "web")
if os.path.isdir(_web_dir):
    app.mount("/static", StaticFiles(directory=_web_dir), name="static")

# --- Models ------------------------------------------------------------------
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    n_results: int = Field(default=5, ge=1, le=15)
    session_id: Optional[str] = None
    stream: bool = False

class ChatResponse(BaseModel):
    status: str
    query: str
    response: str
    session_id: Optional[str] = None
    sources: list = []
    processing_time: float = 0.0
    cached: bool = False

class SearchRequest(BaseModel):
    state: Optional[str] = None
    district: Optional[str] = None
    category: Optional[str] = None
    min_stage: Optional[float] = None
    max_stage: Optional[float] = None

# --- Gemini call -------------------------------------------------------------
GEMINI_URL = (
    "https://generativelanguage.googleapis.com"
    "/v1beta/models/gemini-2.5-flash:generateContent"
)

SYSTEM_PROMPT = """You are INGRES-AI, an expert assistant for India Groundwater Resource Estimation System.
Answer questions about groundwater availability, extraction rates, and district classifications using CGWB data.

Classification:
- Safe           : Stage of GW Development < 70%
- Semi-Critical  : 70% to 90%
- Critical       : 90% to 100%
- Over-Exploited : > 100%

Rules:
1. Cite specific numbers (ham = hectare-meters, bcm = billion cubic meters).
2. If context lacks the answer, say so clearly.
3. Use markdown bullets for multiple items.
4. Keep answers under 250 words unless asked for detail.
5. End with a one-line Insight: summary."""

def _build_prompt(query: str, context: str) -> str:
    return f"""{SYSTEM_PROMPT}

---
RETRIEVED DATA:
{context}
---

QUESTION: {query}

ANSWER:"""

def _call_gemini(prompt: str, history: list = None, timeout: int = 45) -> str:
    if not api_key:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY not configured. Set it in Railway Variables.")

    contents = []
    if history:
        for msg in history[-6:]:
            contents.append({"role": msg["role"], "parts": [{"text": msg["content"]}]})
    contents.append({"role": "user", "parts": [{"text": prompt}]})

    payload = {
        "contents": contents,
        "generationConfig": {"temperature": 0.2, "topP": 0.9, "maxOutputTokens": 1024},
    }

    last_err = "unknown"
    # Only models confirmed available for free-tier API keys
    # gemini-1.5-flash is NOT listed in your account's available models
    models_to_try = [
        "gemini-2.5-flash",       # Primary (confirmed working)
        "gemini-2.0-flash",       # Fallback (confirmed in ListModels)
        "gemini-2.0-flash-lite",  # Lightest fallback - highest rate limits
    ]
    base_url = "https://generativelanguage.googleapis.com/v1beta/models"

    for model in models_to_try:
        for attempt in range(3):  # 3 attempts per model
            try:
                url = f"{base_url}/{model}:generateContent?key={api_key}"
                resp = requests.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=timeout,
                )
                if resp.status_code == 200:
                    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                elif resp.status_code == 404:
                    last_err = f"{model} not available on this API key"
                    break  # try next model
                elif resp.status_code == 429:
                    wait = min(int(resp.headers.get("Retry-After", 10 * (attempt + 1))), 30)
                    last_err = f"{model} rate-limited (429)"
                    time.sleep(wait)
                elif resp.status_code == 503:
                    last_err = f"{model} overloaded (503)"
                    time.sleep(5 * (attempt + 1))
                elif resp.status_code in (400, 401, 403):
                    try:
                        err_body = resp.json()
                        err_msg = err_body.get("error", {}).get("message", resp.text[:200])
                    except Exception:
                        err_msg = resp.text[:200]
                    raise HTTPException(
                        status_code=401,
                        detail=f"Gemini API key error: {err_msg}. "
                               "Update GEMINI_API_KEY in Railway Variables."
                    )
                else:
                    last_err = f"{model} HTTP {resp.status_code}"
                    break
            except requests.Timeout:
                last_err = f"{model} timed out"
                time.sleep(3)
            except requests.RequestException as e:
                last_err = f"{model} connection error: {str(e)[:60]}"
                break

    raise HTTPException(
        status_code=503,
        detail=f"All AI models unavailable. Last: {last_err}. "
               "Free tier limit: 5 req/min — wait 1 minute and try again."
    )

# --- Endpoints ---------------------------------------------------------------
@app.get("/", tags=["Meta"], include_in_schema=False)
async def root():
    """Serve the frontend web app."""
    index_path = os.path.join(_project_root, "web", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    # Fallback JSON if frontend not present
    return {"status": "running", "version": "2.0.0",
            "docs": "/docs", "health": "/health"}

@app.get("/health", tags=["Meta"])
async def health():
    return {
        "status": "healthy",
        "vector_store": collection.count() if collection else "unavailable",
        "csv_rows": len(df_groundwater) if df_groundwater is not None else "unavailable",
        "cache_entries": len(_response_cache),
        "timestamp": datetime.utcnow().isoformat(),
    }

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    t0 = time.time()
    query = request.query.strip()

    if collection is None and df_groundwater is None:
        raise HTTPException(status_code=503,
            detail="No data loaded. Run the embedding notebooks first.")

    # Cache check (only for anonymous queries)
    ck = _cache_key(query)
    if not request.session_id:
        hit = _get_cache(ck)
        if hit:
            return ChatResponse(status="success", query=query, response=hit,
                                processing_time=round(time.time()-t0, 3), cached=True)

    # Vector search
    context, sources = "", []
    if collection:
        try:
            n = min(request.n_results, collection.count())
            res = collection.query(query_texts=[query], n_results=n)
            context = "\n\n".join(res.get("documents", [[]])[0])
            sources = [m.get("source", "") for m in res.get("metadatas", [[]])[0]
                       if isinstance(m, dict) and m.get("source")]
        except Exception as e:
            context = f"[Vector search error: {e}]"
    elif df_groundwater is not None:
        q_lower = query.lower()
        # Try keyword match on state/district first
        mask = df_groundwater.apply(
            lambda r: q_lower in str(r["state"]).lower()
                      or q_lower in str(r["district"]).lower(), axis=1)
        rows = df_groundwater[mask].head(10)

        if rows.empty:
            # No keyword match — check for category keywords in query
            if any(w in q_lower for w in ("over-exploit", "overexploit", "critical", "urgent", "stress", "worst", "highest")):
                rows = df_groundwater.nlargest(15, "stage_pct")[
                    ["state", "district", "stage_pct", "category"]]
            elif any(w in q_lower for w in ("safe", "good", "best", "lowest")):
                rows = df_groundwater.nsmallest(15, "stage_pct")[
                    ["state", "district", "stage_pct", "category"]]
            else:
                # General question — send a representative sample with all categories
                rows = df_groundwater.nlargest(20, "stage_pct")[
                    ["state", "district", "stage_pct", "category"]]

        context = rows.to_string(index=False) if not rows.empty else "No matching records."

    prompt = _build_prompt(query, context)
    history = _conversations.get(request.session_id, []) if request.session_id else []
    answer = _call_gemini(prompt, history)

    if request.session_id:
        _conversations.setdefault(request.session_id, [])
        _conversations[request.session_id] += [
            {"role": "user", "content": query},
            {"role": "model", "content": answer},
        ]
        _conversations[request.session_id] = _conversations[request.session_id][-20:]
    else:
        _set_cache(ck, answer)

    return ChatResponse(
        status="success", query=query, response=answer,
        session_id=request.session_id,
        sources=list(set(sources))[:3],
        processing_time=round(time.time()-t0, 3),
        cached=False,
    )

@app.get("/api/stats", tags=["Data"])
async def get_stats():
    if df_groundwater is None:
        raise HTTPException(status_code=503, detail="CSV data not loaded")
    df = df_groundwater
    cats = df["category"].value_counts().to_dict()
    state_avg = (df.groupby("state")["stage_pct"].mean().round(1)
                 .sort_values(ascending=False).head(15).to_dict())
    top_stressed = (df.nlargest(5, "stage_pct")
                    [["state", "district", "stage_pct", "category"]]
                    .to_dict(orient="records"))
    buckets = {"<30%": 0, "30-50%": 0, "50-70%": 0,
               "70-90%": 0, "90-100%": 0, ">100%": 0}
    for v in df["stage_pct"]:
        if v < 30:      buckets["<30%"] += 1
        elif v < 50:    buckets["30-50%"] += 1
        elif v < 70:    buckets["50-70%"] += 1
        elif v < 90:    buckets["70-90%"] += 1
        elif v <= 100:  buckets["90-100%"] += 1
        else:           buckets[">100%"] += 1
    return {
        "total_districts": len(df),
        "total_states": int(df["state"].nunique()),
        "categories": cats,
        "state_avg_stage": state_avg,
        "top_stressed_districts": top_stressed,
        "stage_distribution": buckets,
        "national_avg_stage": round(float(df["stage_pct"].mean()), 1),
    }

@app.post("/api/search", tags=["Data"])
async def search_data(req: SearchRequest):
    if df_groundwater is None:
        raise HTTPException(status_code=503, detail="CSV data not loaded")
    df = df_groundwater.copy()
    if req.state:
        df = df[df["state"].str.contains(req.state, case=False, na=False)]
    if req.district:
        df = df[df["district"].str.contains(req.district, case=False, na=False)]
    if req.category:
        df = df[df["category"].str.lower() == req.category.lower()]
    if req.min_stage is not None:
        df = df[df["stage_pct"] >= req.min_stage]
    if req.max_stage is not None:
        df = df[df["stage_pct"] <= req.max_stage]
    records = df.head(50).fillna("N/A").to_dict(orient="records")
    return {"count": len(df), "results": records}

@app.delete("/api/history/{session_id}", tags=["Chat"])
async def clear_history(session_id: str):
    _conversations.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}

# --- Run ---------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 40)
    print("  INGRES CHATBOT API SERVER v2.0")
    print("=" * 40 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")