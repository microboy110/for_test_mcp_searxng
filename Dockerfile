# Docker 构建模式（备用）
FROM python:3.11-slim

WORKDIR /app

# 依赖安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 应用代码
COPY . .

# Zeabur 自动注入 PORT
EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
