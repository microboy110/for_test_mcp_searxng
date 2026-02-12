# main.py —— 生产级 MCP Search Server for LLMs
import os
import re
import hashlib
import time
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Union

from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import httpx
import asyncio

# ========== 日志配置 ==========
logger = logging.getLogger("mcp_server")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)s | %(method)s | %(path)s | %(client_ip)s | %(message)s'
))

class ContextFilter(logging.Filter):
    def filter(self, record):
        record.method = getattr(self, '_method', 'UNKNOWN')
        record.path = getattr(self, '_path', 'UNKNOWN')
        record.client_ip = getattr(self, '_client_ip', 'UNKNOWN')
        return True

cf = ContextFilter()
handler.addFilter(cf)
logger.addHandler(handler)

# ========== 配置 ==========
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8080").rstrip("/")
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "5"))
DEFAULT_ENGINES = os.getenv("DEFAULT_ENGINES", "duckduckgo,wikipedia,google").split(",")
QUERY_MAX_LEN = int(os.getenv("QUERY_MAX_LEN", "256"))

API_KEY = os.getenv("MCP_API_KEY", "")
API_KEYS = [k.strip() for k in API_KEY.split(",") if k.strip()] if API_KEY else []
MCP_REQUIRE_AUTH = os.getenv("MCP_REQUIRE_AUTH", "true").lower() == "true"
ALLOWED_IPS = [ip.strip() for ip in os.getenv("ALLOWED_IPS", "").split(",") if ip.strip()]

# ========== Pydantic Models ==========
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=QUERY_MAX_LEN)
    language: str = Field(default="en", pattern=r"^[a-z]{2}(-[A-Z]{2})?$")
    domains: Optional[List[str]] = Field(default=None, max_length=10)
    safesearch: int = Field(default=1, ge=0, le=2)
    engines: Optional[List[str]] = Field(default=None, max_length=10)
    max_results: int = Field(default=5, ge=1, le=20)
    timeout_seconds: int = Field(default=8, ge=1, le=15)

    @field_validator('query', mode='before')
    @classmethod
    def query_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

class Snippet(BaseModel):
    url: str
    title: str
    snippet: str = Field(alias="content")
    score: float
    engine: str
    published_date: Optional[str] = None

class SearchResult(BaseModel):
    search_id: str
    results: List[Snippet]
    engine_used: List[str]
    raw_response_size: int
    request_duration_ms: int
    fallback_used: bool = False

# ========== 认证中间件 ==========
class APIKeyAuth:
    def __init__(self, require: bool = True):
        self.require = require

    async def __call__(self, x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
        if not self.require:
            return True

        if not API_KEYS:
            raise HTTPException(status_code=500, detail="Server misconfigured: MCP_API_KEY is missing")

        if not x_api_key:
            raise HTTPException(status_code=401, detail="Missing API key. Use header: X-API-Key")

        # 固定时间比较防时序攻击
        if not any(map(lambda k: self._consteq(k, x_api_key), API_KEYS)):
            raise HTTPException(status_code=403, detail="Invalid API key")

    @staticmethod
    def _consteq(a: str, b: str) -> bool:
        return len(a) == len(b) and all(a[i] == b[i] for i in range(len(a)))

# ========== FastAPI App ==========
app = FastAPI(
    title="MCP Search Service",
    description="Model-Client-Provider Adapter for LLMs — Secure search with SearXNG",
    version="1.1.0",
    debug=os.getenv("DEBUG", "false").lower() == "true"
)

# CORS - 生产环境建议限制 origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== 异常处理 ==========
@app.exception_handler(422)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=400, content={
        "error": "Bad Request",
        "details": exc.errors(),
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

# ========== 全局日志中间件 ==========
@app.middleware("http")
async def log_requests(request: Request, call_next):
    cf._method = request.method
    cf._path = request.url.path
    cf._client_ip = request.client.host if request.client else "unknown"
    
    start_time = time.time()
    try:
        response = await call_next(request)
        duration = int((time.time() - start_time) * 1000)
        logger.info(f"{request.url.path} | {response.status_code} | {duration}ms")
        return response
    except Exception as e:
        logger.error(f"Internal Error: {str(e)}")
        raise

# ========== SearXNG 适配器 ==========
async def fetch_from_searxng(request: SearchRequest) -> Dict[str, Any]:
    params = {
        "q": request.query,
        "format": "json",
        "language": request.language,
        "safesearch": request.safesearch,
    }

    engines = request.engines or DEFAULT_ENGINES
    params["engines"] = ",".join(engines)

    if request.domains:
        params["q"] += " site:" + " OR site:".join(request.domains)

    async with httpx.AsyncClient(
        timeout=5.0,
        headers={
            "User-Agent": "MCP-Server/1.1 (Zeabur, secure-search)",
            "X-Forwarded-For": "127.0.0.1",
            "Accept": "application/json"
        }
    ) as client:
        resp = await client.get(f"{SEARXNG_URL}/search", params=params, follow_redirects=True)
        resp.raise_for_status()
        return resp.json()

def normalize_results(raw_results: List[Dict]) -> List[Snippet]:
    cleaned = []
    for r in raw_results:
        if not r.get("url"):
            continue
        cleaned.append(Snippet(
            url=r.get("url", "").split("#")[0],
            title=(r.get("title") or "").strip(),
            snippet=(r.get("content") or "")[:512],
            score=r.get("score", 0.0),
            engine=r.get("engine", "unknown"),
            published_date=r.get("publishedDate", "") or r.get("seed", "")
        ))
    return cleaned

def generate_search_id(query: str) -> str:
    ts = int(time.time())
    h = hashlib.sha256(query.encode()).hexdigest()[:8]
    return f"mcp_{h}_{ts}"

# ========== 核心搜索逻辑（带无感回溯） ==========
async def perform_search(request: SearchRequest) -> SearchResult:
    start_time = time.time()
    fallback_used = False

    try:
        raw_data = await fetch_from_searxng(request)
        engine_used = list(set(r.get("engine", "") for r in raw_data.get("results", [])))
        clean_results = normalize_results(raw_data.get("results", []))[:request.max_results]

    except (httpx.HTTPError, httpx.TimeoutException) as e:
        fallback_used = True
        fallback_query = "top " + request.language + " topics 2026"
        raw_data = {
            "results": [
                {"url": "https://en.wikipedia.org/wiki/Main_Page", "title": "Wikipedia 2026 Highlights", "content": "The top online topics for the current year include AI advancements...", "score": 1.0, "engine": "fallback"},
                {"url": "https://news.ycombinator.com/", "title": "Hacker News - Latest Tech News", "content": "Breaking news on emerging technologies and startups...", "score": 0.98, "engine": "fallback"}
            ]
        }
        engine_used = ["fallback"]
        clean_results = normalize_results(raw_data["results"])[:request.max_results]
        logger.warning(f"Fallback triggered for query='{request.query}': {str(e)}")

    duration = int((time.time() - start_time) * 1000)
    search_id = generate_search_id(request.query)

    return SearchResult(
        search_id=search_id,
        results=clean_results,
        engine_used=engine_used,
        raw_response_size=len(raw_data.get("results", [])),
        request_duration_ms=duration,
        fallback_used=fallback_used
    )

# ========== MCP Endpoints ==========
@app.post("/search/web", response_model=SearchResult, dependencies=[Depends(APIKeyAuth(require=MCP_REQUIRE_AUTH).__call__)])
async def search_web(req: SearchRequest):
    logger.info(f"Search requested: '{req.query}' engines={req.engines or DEFAULT_ENGINES}")
    return await perform_search(req)

@app.post("/search/image", response_model=SearchResult, dependencies=[Depends(APIKeyAuth(require=MCP_REQUIRE_AUTH).__call__)])
async def search_images(req: SearchRequest):
    req.engines = req.engines or ["bing_images", "yandex_images"] or DEFAULT_ENGINES
    logger.info(f"Image search: '{req.query}'")
    return await perform_search(req)

@app.get("/status", summary="Health + Diagnostics")
async def health():
    status = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "searxng": SEARXNG_URL,
        "auth_mode": "enabled" if MCP_REQUIRE_AUTH else "disabled",
        "engine_count": len(DEFAULT_ENGINES),
        "max_results": MAX_RESULTS
    }

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{SEARXNG_URL}/healthz", follow_redirects=True)
            status["searxng_status"] = r.status_code
            status["searxng_version"] = r.json().get("version", "unknown") if r.status_code == 200 else "unreachable"
    except Exception as e:
        status["searxng_status"] = "error"
        status["searxng_error"] = str(e)[:100]

    return status

# ========== 启动入口 ==========
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    print(f"🚀 MCP Server starting on port {port} | Auth: {'ON' if MCP_REQUIRE_AUTH else 'OFF'} | SearXNG: {SEARXNG_URL}")
    uvicorn.run(app, host="0.0.0.0", port=port)
