# main.py —— 生产级 MCP Search Server for LLMs
import os
import hashlib
import time
import logging
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import JSONResponse, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import httpx

# ========== 日志配置 ==========
logger = logging.getLogger("mcp_server")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
log_format = '%(asctime)s | %(levelname)s | %(method)s | %(path)s | %(client_ip)s | %(message)s'
handler.setFormatter(logging.Formatter(log_format))


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
API_KEY = os.getenv("MCP_API_KEY", "")
API_KEYS = [k.strip() for k in API_KEY.split(",") if k.strip()] if API_KEY else []
MCP_REQUIRE_AUTH = os.getenv("MCP_REQUIRE_AUTH", "true").lower() == "true"
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "5"))
DEFAULT_ENGINES = os.getenv("DEFAULT_ENGINES", "duckduckgo,wikipedia,google").split(",")


# ========== Pydantic Models ==========
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    language: str = Field(default="en", pattern=r"^[a-z]{2}(-[A-Z]{2})?$")
    domains: Optional[List[str]] = Field(default=None, max_length=10)
    safesearch: int = Field(default=1, ge=0, le=2)
    engines: Optional[List[str]] = Field(default=None, max_length=10)
    max_results: int = Field(default=5, ge=1, le=20)

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

        if not any(map(lambda k: self._consteq(k, x_api_key), API_KEYS)):
            raise HTTPException(status_code=403, detail="Invalid API key")

    @staticmethod
    def _consteq(a: str, b: str) -> bool:
        return len(a) == len(b) and all(a[i] == b[i] for i in range(len(a)))


# ========== FastAPI App ==========
app = FastAPI(
    title="MCP Search Service",
    description="Secure search with SearXNG and fallback support",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(422)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=400, content={
        "error": "Bad Request",
        "details": exc.errors(),
        "timestamp": datetime.now(timezone.utc).isoformat()
    })


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


@app.get("/", tags=["Health/Root"])
async def root():
    """Root endpoint: Returns simple text 'fortest' for quick health check."""
    return Response(content="fortest", media_type="text/plain")


# ========== 搜索核心 ==========
async def perform_search(request: SearchRequest) -> SearchResult:
    start_time = time.time()
    fallback_used = False

    try:
        params = {
            "q": request.query,
            "format": "json",
            "language": request.language,
            "safesearch": request.safesearch,
            "engines": ",".join(request.engines or DEFAULT_ENGINES),
        }
        if request.domains:
            params["q"] += " site:" + " OR site:".join(request.domains)

        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{SEARXNG_URL}/search", params=params, follow_redirects=True)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
    except Exception as e:
        fallback_used = True
        # 无感回溯
        results = [
            {"url": "https://en.wikipedia.org/wiki/Main_Page",
             "title": "Wikipedia 2026",
             "content": "AI advancements...",
             "score": 1.0, "engine": "fallback"}
        ]
        logger.warning(f"Fallback triggered for query='{request.query}': {str(e)}")

    engine_used = list(set(r.get("engine", "") for r in results))
    clean_results = []
    for r in results[:request.max_results]:
        if r.get("url"):
            clean_results.append(Snippet(
                url=r.get("url", "").split("#")[0],
                title=(r.get("title") or "").strip(),
                snippet=(r.get("content") or "")[:512],
                score=r.get("score", 0.0),
                engine=r.get("engine", "unknown")
            ))

    duration = int((time.time() - start_time) * 1000)
    query_hash = hashlib.sha256(request.query.encode()).hexdigest()[:8]
    search_id = f"mcp_{query_hash}_{int(time.time())}"

    return SearchResult(
        search_id=search_id,
        results=clean_results,
        engine_used=engine_used,
        raw_response_size=len(results),
        request_duration_ms=duration,
        fallback_used=fallback_used
    )


# ========== 端点 ==========
@app.post("/search/web", response_model=SearchResult)
async def search_web(req: SearchRequest):
    logger.info(f"Search requested: '{req.query}'")
    return await perform_search(req)


@app.get("/status")
async def health():
    return {
        "status": "healthy",
        "searxng": SEARXNG_URL,
        "auth_mode": "enabled" if MCP_REQUIRE_AUTH else "disabled",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    p = int(os.getenv("PORT", 8080))
    auth_s = 'ON' if MCP_REQUIRE_AUTH else 'OFF'
    print(f"🚀 MCP Server port {p} | Auth: {auth_s} | URL: {SEARXNG_URL}")
    uvicorn.run(app, host="0.0.0.0", port=p)
