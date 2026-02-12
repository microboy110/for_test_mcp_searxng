#!/bin/bash
# deploy.sh —— Zeabur 一键部署（需提前安装 Zeabur CLI）
set -e

echo "🚀 Deploying MCP Server to Zeabur..."

# 1. 创建服务（若不存在）
if ! zeabur service ls | grep -q "mcp-server"; then
  zeabur service create mcp-server --template python
fi

# 2. 设置环境变量（若未设置）
zeabur service env set --service mcp-server \
  SEARXNG_URL="${SEARXNG_URL:-https://searxng.example.com}" \
  MCP_API_KEY="${MCP_API_KEY:-$(python -c 'import secrets;print(secrets.token_urlsafe(32))')}" \
  MCP_REQUIRE_AUTH="${MCP_REQUIRE_AUTH:-true}"

# 3. 部署（自动 Git Push）
zeabur deploy --service mcp-server --url https://github.com/${GITHUB_USER:-yourname}/mcp-server-zeabur.git

echo "✅ Deployed! Access https://$(zeabur service ls | grep mcp-server | awk '{print $3}')"
