# -*- coding: utf-8 -*-
"""
FastAPI Server for INGRES Chatbot - Production v3.0 (Optimized)
================================================================
Performance design:
  - CSV loaded ONCE at startup from pre-baked groundwater_clean.csv
  - Pandas index on state/district for O(1) lookups
  - Stats pre-computed at startup — /api/stats is instant
  - LRU response cache (200 entries, MD5 keyed)
  - Async Gemini HTTP call (non-blocking event loop)
  - Smart query routing — only sends filtered data to LLM
  - Full graceful degradation if Gemini is unavailable
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import sys, os, time, hashlib, asyncio, logging
from collections import OrderedDict
from datetime import datetime
import httpx  # async HTTP — replaces requests for Gemini calls

# ── Project root ────────────────────────────────────────────────────────────
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dotenv import load_dotenv
load_dotenv()

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ingres")

# ── Optional deps ────────────────────────────────────────────────────────────
try:
    import pandas as pd
    PANDAS_OK = True
except ImportError:
    PANDAS_OK = False
    log.warning("pandas not installed — CSV features disabled")

try:
    import chromadb
    CHROMA_OK = True
except Exception:
    CHROMA_OK = False

# ── Global state ─────────────────────────────────────────────────────────────
collection      = None
api_key: str    = ""
df_gw           = None          # main DataFrame (indexed)
_stats_cache    = None          # pre-computed stats dict (built at startup)
_CACHE_MAX      = 300
_response_cache: OrderedDict = OrderedDict()
_conversations:  dict        = {}

# ── Helpers ───────────────────────────────────────────────────────────────────
def _categorize(stage: float) -> str:
    if stage < 70:    return "Safe"
    elif stage < 90:  return "Semi-Critical"
    elif stage <= 100: return "Critical"
    else:             return "Over-Exploited"

def _load_df(path: str) -> "pd.DataFrame":
    """
    Load the pre-baked clean CSV produced by scripts/preprocess.py.
    Falls back to parsing the raw CSV if the clean one isn't found.
    """
    df = pd.read_csv(path)
    df["state"]    = df["state"].str.strip().str.title()
    df["district"] = df["district"].str.strip().str.title()
    df["stage_pct"] = pd.to_numeric(df["stage_pct"], errors="coerce")
    df = df.dropna(subset=["stage_pct"])
    if "category" not in df.columns:
        df["category"] = df["stage_pct"].apply(_categorize)
    # Set index for fast lookups
    df = df.set_index(["state", "district"], drop=False)
    df.index.names = ["_state_idx", "_dist_idx"]
    return df

def _build_stats(df: "pd.DataFrame") -> Dict[str, Any]:
    """Pre-compute all stats once at startup."""
    cats = df["category"].value_counts().to_dict()
    state_avg = (
        df.groupby("state")["stage_pct"].mean()
        .round(1).sort_values(ascending=False).head(15).to_dict()
    )
    top_stressed = (
        df.nlargest(5, "stage_pct")
        [["state", "district", "stage_pct", "category"]]
        .to_dict(orient="records")
    )
    # Stage distribution buckets — vectorized (no Python loop)
    s = df["stage_pct"]
    buckets = {
        "<30%":    int((s < 30).sum()),
        "30-50%":  int(((s >= 30) & (s < 50)).sum()),
        "50-70%":  int(((s >= 50) & (s < 70)).sum()),
        "70-90%":  int(((s >= 70) & (s < 90)).sum()),
        "90-100%": int(((s >= 90) & (s <= 100)).sum()),
        ">100%":   int((s > 100).sum()),
    }
    return {
        "total_districts":       int(len(df)),
        "total_states":          int(df["state"].nunique()),
        "categories":            cats,
        "state_avg_stage":       state_avg,
        "top_stressed_districts":top_stressed,
        "stage_distribution":    buckets,
        "national_avg_stage":    round(float(s.mean()), 1),
    }

def _query_df(df: "pd.DataFrame", query: str) -> "pd.DataFrame":
    """
    Fast pandas query routing — returns the most relevant rows for a query.
    Uses vectorized string operations (no .apply() row loop).
    """
    q = query.lower().strip()

    # ── State abbreviation expansion ────────────────────────────────────────
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

    # 1. State match — return ALL districts for that state
    state_mask = df["state"].str.lower().str.contains(q, na=False, regex=False)
    if state_mask.any():
        state_rows = df[state_mask]
        # If we matched exactly one state, return all its districts
        matched_states = state_rows["state"].unique()
        if len(matched_states) <= 2:
            return state_rows  # return ALL districts of matched state(s)
        return state_rows.head(30)

    # 2. District match
    dist_mask = df["district"].str.lower().str.contains(q, na=False, regex=False)
    if dist_mask.any():
        return df[dist_mask].head(12)

    # 3. Category keyword routing
    if any(w in q for w in ("over-exploit", "overexploit", "critical", "urgent", "stress", "worst", "highest", "dangerous")):
        return df.nlargest(20, "stage_pct")[["state", "district", "stage_pct", "category"]]
    if any(w in q for w in ("safe", "good", "best", "lowest", "healthy")):
        return df.nsmallest(20, "stage_pct")[["state", "district", "stage_pct", "category"]]
    if any(w in q for w in ("semi", "moderate")):
        return df[df["category"] == "Semi-Critical"].head(20)
    if any(w in q for w in ("all state", "every state", "list state", "how many state")):
        # Return one row per state (highest stage)
        return df.sort_values("stage_pct", ascending=False).drop_duplicates("state")[["state", "district", "stage_pct", "category"]]

    # 4. General question — top stressed districts across all states
    return df.nlargest(25, "stage_pct")[["state", "district", "stage_pct", "category"]]

# ── Cache ────────────────────────────────────────────────────────────────────
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

# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global collection, api_key, df_gw, _stats_cache

    log.info("=== INGRES Chatbot API v3 Starting ===")
    t0 = time.time()

    # 1. Gemini API key
    api_key = os.getenv("GEMINI_API_KEY", "")
    if api_key:
        log.info("Gemini key loaded OK")
    else:
        log.warning("GEMINI_API_KEY not set — chat will fail")

    # 2. ChromaDB (optional vector store)
    if CHROMA_OK:
        try:
            client = chromadb.PersistentClient(
                path=os.path.join(_project_root, "data", "embeddings")
            )
            collection = client.get_collection("groundwater_data")
            log.info(f"Vector store loaded: {collection.count()} docs")
        except Exception as e:
            log.warning(f"Vector store unavailable: {e}")

    # 3. Load pre-baked clean CSV (fast — simple pd.read_csv)
    if PANDAS_OK:
        clean_csv = os.path.join(_project_root, "data", "processed", "groundwater_clean.csv")
        raw_csv   = os.path.join(_project_root, "data", "processed", "groundwater_data.csv")

        csv_path = clean_csv if os.path.exists(clean_csv) else raw_csv
        if os.path.exists(csv_path):
            try:
                df_gw = _load_df(csv_path)
                _stats_cache = _build_stats(df_gw)
                log.info(f"CSV loaded: {len(df_gw)} districts | used: {os.path.basename(csv_path)}")
            except Exception as e:
                log.error(f"CSV load failed: {e}")
        else:
            log.warning("No CSV found — data endpoints disabled")

    log.info(f"Startup complete in {time.time()-t0:.2f}s")
    yield
    log.info("Shutting down.")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="INGRES Chatbot API",
    description="High-performance AI chatbot for India Groundwater Resource Estimation System",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend
_web_dir = os.path.join(_project_root, "web")
if os.path.isdir(_web_dir):
    app.mount("/static", StaticFiles(directory=_web_dir), name="static")

# ── Pydantic models ───────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query:      str   = Field(..., min_length=1, max_length=1000)
    n_results:  int   = Field(default=5, ge=1, le=15)
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    status:          str
    query:           str
    response:        str
    session_id:      Optional[str] = None
    sources:         list = []
    processing_time: float = 0.0
    cached:          bool  = False

class SearchRequest(BaseModel):
    state:     Optional[str]   = None
    district:  Optional[str]   = None
    category:  Optional[str]   = None
    min_stage: Optional[float] = None
    max_stage: Optional[float] = None

# ── Gemini (async) ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are INGRES-AI, an expert assistant for India Groundwater Resource Estimation System.
Answer questions about groundwater availability, extraction rates, and district classifications using CGWB data.

Classification:
- Safe           : Stage of GW Development < 70%
- Semi-Critical  : 70% to 90%
- Critical       : 90% to 100%
- Over-Exploited : > 100%

Rules:
1. Cite specific numbers (ham = hectare-meters, bcm = billion cubic meters).
2. If context lacks the answer, say so clearly — do NOT hallucinate.
3. Use markdown bullets for multiple items.
4. Keep answers under 250 words unless asked for detail.
5. End with a one-line Insight: summary."""

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
MODELS      = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]  # verified working

async def _call_gemini_async(prompt: str, history: list = None, timeout: int = 25) -> str:
    """Async Gemini call — fail-fast strategy to avoid UI timeouts."""
    if not api_key:
        raise HTTPException(503, "GEMINI_API_KEY not configured.")

    contents = []
    if history:
        for msg in history[-6:]:
            contents.append({"role": msg["role"], "parts": [{"text": msg["content"]}]})
    contents.append({"role": "user", "parts": [{"text": prompt}]})

    payload = {
        "contents": contents,
        "generationConfig": {"temperature": 0.2, "topP": 0.9, "maxOutputTokens": 600},
    }

    # Try primary then fallback — both verified working
    MODELS_FAST = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
    last_err = "unknown"

    async with httpx.AsyncClient(timeout=timeout) as client:
        for model in MODELS_FAST:
            for attempt in range(2):
                try:
                    url  = f"{GEMINI_BASE}/{model}:generateContent?key={api_key}"
                    resp = await client.post(url, json=payload,
                                             headers={"Content-Type": "application/json"})
                    if resp.status_code == 200:
                        log.info(f"Gemini OK [{model}] attempt={attempt+1}")
                        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                    elif resp.status_code == 429:
                        last_err = f"{model} rate-limited (429)"
                        log.warning(f"{model} rate-limited — short wait")
                        await asyncio.sleep(3)   # Short wait only — don't block for 30s
                    elif resp.status_code in (400, 401, 403):
                        try:
                            err_msg = resp.json().get("error", {}).get("message", "")
                        except Exception:
                            err_msg = resp.text[:200]
                        raise HTTPException(401, f"Gemini API key error: {err_msg}.")
                    elif resp.status_code == 404:
                        last_err = f"{model} not available"
                        break   # Try next model immediately
                    else:
                        last_err = f"{model} HTTP {resp.status_code}"
                        break
                except httpx.TimeoutException:
                    last_err = f"{model} timed out"
                    log.warning(f"{model} timeout (attempt {attempt+1})")
                    break   # Don't retry on timeout — try next model
                except httpx.RequestError as e:
                    last_err = f"{model} connection error"
                    log.error(f"{model} request error: {e}")
                    break

    raise HTTPException(
        503,
        f"AI service temporarily unavailable. Last error: {last_err}. "
        "Free tier: 15 req/min — please wait a moment and try again."
    )

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    index_path = os.path.join(_project_root, "web", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return {"status": "running", "version": "3.0.0", "docs": "/docs"}

@app.get("/health", tags=["Meta"])
async def health():
    return {
        "status":        "healthy",
        "version":       "3.0.0",
        "vector_store":  collection.count() if collection else "unavailable",
        "csv_rows":      len(df_gw) if df_gw is not None else 0,
        "cache_entries": len(_response_cache),
        "timestamp":     datetime.utcnow().isoformat(),
    }

@app.get("/api/debug", tags=["Meta"])
async def debug():
    import platform
    key = os.getenv("GEMINI_API_KEY", "")
    clean_exists = os.path.exists(
        os.path.join(_project_root, "data", "processed", "groundwater_clean.csv"))
    return {
        "gemini_key_set":      bool(key),
        "gemini_key_prefix":   key[:8] + "..." if key else "NOT SET",
        "csv_loaded":          df_gw is not None,
        "csv_rows":            len(df_gw) if df_gw is not None else 0,
        "clean_csv_exists":    clean_exists,
        "vector_store":        collection.count() if collection else "unavailable",
        "stats_precomputed":   _stats_cache is not None,
        "python_version":      platform.python_version(),
        "env_port":            os.getenv("PORT", "not set (local)"),
        "web_dir_exists":      os.path.isdir(os.path.join(_project_root, "web")),
    }

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    t0 = time.time()
    query = request.query.strip()

    if df_gw is None and collection is None:
        raise HTTPException(503, "No data loaded — check Railway logs.")

    # Cache hit (anonymous queries only)
    ck = _cache_key(query)
    if not request.session_id:
        cached = _get_cache(ck)
        if cached:
            return ChatResponse(
                status="success", query=query, response=cached,
                processing_time=round(time.time()-t0, 3), cached=True
            )

    # Build context — CSV is ALWAYS the primary source (634 districts, all states)
    # Vector store supplements only if it has valid district documents
    context, sources = "", []

    # 1. Always query the structured CSV first (guaranteed to have all districts)
    if df_gw is not None:
        rows = _query_df(df_gw, query)
        context = rows.to_string(index=False) if not rows.empty else "No matching records."

    # 2. Optionally supplement with vector store (only if it has real district docs)
    if collection:
        try:
            n = min(request.n_results, collection.count())
            res = collection.query(query_texts=[query], n_results=n)
            vec_docs = res.get("documents", [[]])[0]
            vec_meta = res.get("metadatas", [[]])[0]
            # Only use vector docs that look like real district entries (have 'state' in metadata)
            valid_docs = [
                doc for doc, meta in zip(vec_docs, vec_meta)
                if isinstance(meta, dict) and meta.get("state")
            ]
            if valid_docs:
                context = context + "\n\nADDITIONAL CONTEXT:\n" + "\n\n".join(valid_docs)
                sources = [
                    m.get("source", "") for m in vec_meta
                    if isinstance(m, dict) and m.get("source")
                ]
        except Exception as e:
            log.error(f"Vector search error: {e}")

    prompt  = f"{SYSTEM_PROMPT}\n\n---\nDATA:\n{context[:6000]}\n---\n\nQUESTION: {query}\n\nANSWER:"
    history = _conversations.get(request.session_id, []) if request.session_id else []
    answer  = await _call_gemini_async(prompt, history, timeout=25)

    # Update conversation history
    if request.session_id:
        hist = _conversations.setdefault(request.session_id, [])
        hist += [{"role": "user", "content": query},
                 {"role": "model", "content": answer}]
        _conversations[request.session_id] = hist[-20:]
    else:
        _set_cache(ck, answer)

    elapsed = round(time.time() - t0, 3)
    log.info(f"chat OK | {elapsed}s | cached=False | query={query[:60]}")
    return ChatResponse(
        status="success", query=query, response=answer,
        session_id=request.session_id,
        sources=list(set(sources))[:3],
        processing_time=elapsed,
        cached=False,
    )

@app.get("/api/stats", tags=["Data"])
async def get_stats():
    """Returns pre-computed stats — instant, no computation at request time."""
    if _stats_cache is None:
        raise HTTPException(503, "Stats not available — CSV not loaded.")
    return _stats_cache

@app.post("/api/search", tags=["Data"])
async def search_data(req: SearchRequest):
    if df_gw is None:
        raise HTTPException(503, "CSV data not loaded.")
    df = df_gw.copy()
    if req.state:
        df = df[df["state"].str.contains(req.state, case=False, na=False, regex=False)]
    if req.district:
        df = df[df["district"].str.contains(req.district, case=False, na=False, regex=False)]
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

# ── Dev server ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("=" * 40)
    print("  INGRES CHATBOT API v3.0 (Optimized)")
    print("=" * 40)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")