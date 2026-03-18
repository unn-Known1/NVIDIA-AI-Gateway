# NVIDIA AI Gateway

Production-ready OpenAI-compatible proxy for NVIDIA AI with streaming, tool calls, CORS, and SQLite logging.

## Features

- Full OpenAI API compatibility
- Streaming (SSE) support
- Tool calling with parallel execution
- CORS enabled
- SQLite request/response logging
- Token tracking
- Cross-platform: Linux, macOS, Windows
- Standalone binaries available

## Quick Start

### Binary (No Python needed)

Download from [Releases](https://github.com/your-username/NVIDIA-AI-Gateway/releases):

```bash
chmod +x nvidia-ai-gateway
export CUSTOM_API_KEY="nvapi-..."
./nvidia-ai-gateway --port 8080
```

### Python

```bash
pip install -r requirements.txt
python -m gateway --port 8080
```

## Configuration

Set environment variables:

- `CUSTOM_API_KEY` (required): Your NVIDIA API key
- `CUSTOM_BASE_URL`: NVIDIA API base URL (default: https://integrate.api.nvidia.com/v1)
- `CUSTOM_MODEL_ID`: Default model (default: stepfun-ai/step-3.5-flash)
- `GATEWAY_PORT`: Port to listen on (default: 8089)

## API Endpoints

All OpenAI-compatible:

- `POST /v1/chat/completions` (streaming & non-streaming)
- `POST /v1/completions`
- `POST /v1/embeddings`
- `GET /v1/models`
- `GET /gateway/status`
- `GET /gateway/stats`
- `GET /gateway/logs`
- `OPTIONS *` (CORS preflight)

## Usage Example

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-gateway-..."  # From gateway startup banner
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

## License

Apache License 2.0 - see LICENSE file.
