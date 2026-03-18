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

Download from [Releases](https://github.com/unn-Known1/NVIDIA-AI-Gateway/releases):

```bash
chmod +x nvidia-ai-gateway
export CUSTOM_API_KEY="nvapi-..."
./nvidia-ai-gateway --port 8080
```

### Python (Package Installation)

```bash
git clone https://github.com/unn-Known1/NVIDIA-AI-Gateway.git
cd NVIDIA-AI-Gateway
./scripts/install.sh
source venv/bin/activate
python -m gateway --port 8080
```

### Python (Manual)

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

## Troubleshooting

### Virtual Environment Creation Fails with ensurepip Error

**Problem:** In some environments (containers, restricted systems), `python3 -m venv` fails with:
```
Error: Command '['.../venv/bin/python3', '-m', 'ensurepip', '--upgrade', '--default-pip']' returned non-zero exit status 1.
```

**Solution:** This is fixed in the current version. The `install.sh` script now:
1. Creates venv without pip using `--without-pip` flag
2. Manually installs pip via get-pip.py
3. Detects incomplete venvs and repairs them automatically

If you encounter this error, simply re-run:
```bash
./scripts/install.sh
```

### Module 'gateway' Not Found After Installation

If you see `ModuleNotFoundError: No module named 'gateway'`:
```bash
# Reinstall the package in editable mode
source venv/bin/activate
pip install -e .
```

### Port Already in Use

If the port is already occupied, use a different port:
```bash
export GATEWAY_PORT=8081
python -m gateway
```

## Development

### Project Structure
```
nvidia-ai-gateway/
├── src/gateway/          # Main package
│   ├── __init__.py
│   └── __main__.py       # Application entry point
├── scripts/              # Platform-specific installers and launchers
├── .github/workflows/    # CI/CD pipelines
├── pyproject.toml        # Package configuration
├── requirements.txt      # Dependencies
├── nvidia-ai-gateway.py  # Single-file portable version
└── README.md
```

### Building Standalone Executables

```bash
# Linux/macOS
./scripts/build.sh

# Windows
scripts\build.bat
```

### Running Tests

```bash
# Basic connectivity test
export CUSTOM_API_KEY="nvapi-..."
python -m gateway &
curl http://localhost:8080/v1/models
```

## License

Apache License 2.0 - see LICENSE file.
EOF
echo "README.md updated successfully"