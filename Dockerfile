FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git \
    && rm -rf /var/lib/apt/lists/*

COPY . /app/

RUN pip install --no-cache-dir \
    "openenv-core[core]>=0.2.2" \
    "pydantic>=2.0.0" \
    "fastapi>=0.115.0" \
    "uvicorn>=0.24.0" \
    "fastmcp>=3.0.0" \
    "openai>=2.7.2" \
    "requests>=2.31.0" \
    "python-dotenv" \
    "httpx>=0.27.0" \
    "gradio>=5.0.0"

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/app"
ENV ENABLE_WEB_INTERFACE=false

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

CMD ["python", "-c", "import uvicorn; uvicorn.run('server.app:app', host='0.0.0.0', port=7860)"]
