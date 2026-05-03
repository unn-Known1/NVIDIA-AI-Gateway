# NVIDIA AI Gateway

Production-ready OpenAI-compatible proxy for NVIDIA AI with streaming, tool calls, CORS, connection pooling, and resilient background architecture.

## Features

- **Full OpenAI API Compatibility:** Drop-in replacement for OpenAI endpoints.
- **Streaming (SSE) Support:** Reliable real-time chunk streaming.
- **Tool Calling:** Full parallel tool call extraction and bridging.
- **Resilience:** Connection pooling, upstream retries (with jittered backoff), and background log retention.
- **Observability:** Structured JSON logging, SQLite request/response auditing, token tracking, and `/healthz` / `/readyz` probes.
- **Security:** In-flight masking of sensitive API keys within logs and explicit model allow-listing.
- **Cross-platform & Standalone:** Available as zero-dependency Linux and Windows executables, Docker containers, and a pip-installable package.

## Quick Start

### 1. Docker (Recommended)

Run the latest image directly from the GitHub Container Registry:

```bash
docker run -d \
  -p 8080:8089 \
  -e CUSTOM_API_KEY="nvapi-..." \
  -v $(pwd)/data:/app/data \
  ghcr.io/unn-known1/nvidia-ai-gateway:latest
```

### 2. Standalone Binary (No Python needed)

Download from [Releases](https://github.com/unn-Known1/NVIDIA-AI-Gateway/releases):

```bash
chmod +x nvidia-ai-gateway-linux-amd64
export CUSTOM_API_KEY="nvapi-..."
./nvidia-ai-gateway-linux-amd64 --port 8080
```

### 3. Python (Virtual Environment)

```bash
git clone https://github.com/unn-Known1/NVIDIA-AI-Gateway.git
cd NVIDIA-AI-Gateway
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m gateway --port 8080
```

## Configuration

You can configure the gateway using environment variables, an `.env` file in the working directory, or by passing a config file (`--config config.ini`). 

### Core Settings
- `CUSTOM_API_KEY` (required): Your NVIDIA API key (`nvapi-...`).
- `CUSTOM_BASE_URL`: NVIDIA API base URL (default: `https://integrate.api.nvidia.com/v1`).
- `CUSTOM_MODEL_ID`: Default model if none is specified (default: `stepfun-ai/step-3.5-flash`).
- `ALLOWED_MODELS`: Comma-separated list of models the client is allowed to request, or `*` to allow any (default: `*`).
- `GATEWAY_PORT`: Port to listen on (default: `8089`).

### Advanced & Reliability Settings
- `UPSTREAM_RETRIES`: Number of times to retry idempotent upstream requests on 5xx/429 errors (default: `3`).
- `LOG_RETENTION_DAYS`: Background worker cleans up SQLite logs older than this (default: `30`).
- `LOG_REQUEST_BODIES`: Log full request/response bodies to DB (`true`/`false`). Secrets are automatically redacted.

## API Endpoints

All endpoints are fully OpenAI-compatible:

- `POST /v1/chat/completions` (streaming & non-streaming)
- `POST /v1/completions`
- `POST /v1/embeddings`
- `GET /v1/models`

### Gateway specific endpoints:
- `GET /healthz` (liveness probe)
- `GET /readyz` (readiness probe - validates DB & upstream API)
- `GET /gateway/status`
- `GET /gateway/stats`
- `GET /gateway/logs`

## Usage Example

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-gateway-..."  # Grab this from the gateway startup banner!
)

response = client.chat.completions.create(
    model="stepfun-ai/step-3.5-flash",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Development & Architecture

### Architecture Overview
The NVIDIA AI Gateway is designed to be lightweight yet highly concurrent:
- **Web Server:** Uses Flask served via a background `werkzeug` WSGI server, capable of handling SSE streams cleanly.
- **Database:** Uses SQLite with `WAL` (Write-Ahead Logging) mode and thread-local connections to allow concurrent non-blocking reads/writes.
- **Connection Pooling:** A global `requests.Session` pool is maintained for upstream requests, eliminating TLS handshake overhead on every API call.
- **Background Workers:** A dedicated daemon thread (`log_retention_worker`) routinely vacuums the database to prevent unbounded disk usage.

### Project Structure
```
nvidia-ai-gateway/
├── src/gateway/          # Main package
│   ├── __init__.py
│   └── __main__.py       # Application entry point
├── scripts/              # Platform-specific installers and launchers
├── .github/workflows/    # CI/CD pipelines (Automated release builds)
├── pyproject.toml        # Package configuration
└── requirements.txt      # Pinned dependencies
```

### Running Tests

```bash
# Basic connectivity test
export CUSTOM_API_KEY="nvapi-..."
python -m gateway &
curl http://localhost:8080/healthz
curl http://localhost:8080/readyz
```

## Troubleshooting

### Virtual Environment Creation Fails with ensurepip Error
**Problem:** `python3 -m venv` fails with an `ensurepip` error.
**Solution:** Re-run `./scripts/install.sh`. It automatically detects broken environments, uses the `--without-pip` flag, and bootstraps pip manually.

### Port Already in Use
If port 8089 (or your configured port) is occupied:
```bash
export GATEWAY_PORT=8081
python -m gateway
```

## License

Apache License 2.0 - see [LICENSE](LICENSE) file.