import os
import hashlib
import time
import logging
import json
from datetime import datetime, timezone
from typing import List, Optional, Any

from fastapi import FastAPI, HTTPException, Request, Header, Depends
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import httpx

# MCP 核心依赖
from mcp.server import Server
from mcp.types import Tool, TextContent
from mcp.server.fastapi import SseServerTransport

# ========== 配置与日志 ==========
logger = logging.getLogger("mcp_server")
logging.basicConfig(level=logging.INFO)

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8080").rstrip("/")
API_KEY = os.getenv("MCP_API_KEY", "")
API_KEYS = [k.strip() for k in API_KEY.split(",") if k.strip()] if API_KEY else []
MCP_REQUIRE_AUTH = os.getenv("MCP_REQUIRE_AUTH", "true").lower() == "true"
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "5"))
DEFAULT_ENGINES = os.getenv("DEFAULT_ENGINES", "duckduckgo,wikipedia,google").split(",")

# ========== FastAPI 基础 ==========
app = FastAPI(title="Pro MCP Search Server", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== 数据模型 ==========
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    max_results: int = Field(default=5, ge=1, le=20)
    language: str = Field(default="en")
    engines: Optional[List[str]] = None

class Snippet(BaseModel):
    url: str
    title: str
    content: str

# ========== 核心搜索逻辑 (复用) ==========
async def perform_search_logic(query: str, count: int) -> str:
    """供内部调用的统一搜索函数，返回文本格式供 LLM 使用"""
    params = {
        "q": query,
        "format": "json",
        "engines": ",".join(DEFAULT_ENGINES),
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{SEARXNG_URL}/search", params=params)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])[:count]
            
            if not results:
                return "No results found."
                
            formatted = []
            for r in results:
                formatted.append(f"Title: {r.get('title')}\nURL: {r.get('url')}\nContent: {r.get('content', '')[:300]}")
            return "\n\n---\n\n".join(formatted)
    except Exception as e:
        logger.error(f"Search Error: {str(e)}")
        return f"Error occurred while searching: {str(e)}"

# ========== 认证逻辑 ==========
async def verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    if not MCP_REQUIRE_AUTH:
        return
    if not x_api_key or x_api_key not in API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid or Missing API Key")

# ========== 1. 标准 MCP SSE 实现 ==========
mcp_server = Server("mcp-web-search")
sse_transport = SseServerTransport("/messages")

@mcp_server.list_tools()
async def handle_list_tools() -> list[Tool]:
    return [
        Tool(
            name="web_search",
            description="搜索互联网获取实时信息",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                },
                "required": ["query"],
            },
        )
    ]

@mcp_server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> list[TextContent]:
    if name == "web_search":
        query = arguments.get("query", "")
        result_text = await perform_search_logic(query, MAX_RESULTS)
        return [TextContent(type="text", text=result_text)]
    raise ValueError(f"Tool not found: {name}")

@app.get("/sse")
async def handle_sse(request: Request, _=Depends(verify_api_key)):
    """客户端连接入口，加入了 API Key 验证"""
    async with sse_transport.connect_scope(request) as scope:
        await mcp_server.run(
            scope[0], scope[1], mcp_server.create_initialization_options()
        )

@app.post("/messages")
async def handle_messages(request: Request):
    """处理 MCP 消息交互"""
    await sse_transport.handle_post_resource(request)

# ========== 2. 原有 HTTP REST 实现 ==========
@app.post("/search/web", dependencies=[Depends(verify_api_key)])
async def legacy_search_endpoint(req: SearchRequest):
    content = await perform_search_logic(req.query, req.max_results)
    return {"results_text": content}

@app.get("/status")
async def health():
    return {"status": "ok", "mcp_protocol": "sse"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
