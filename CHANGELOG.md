# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-03-18

### Added
- Initial release of NVIDIA AI Gateway
- Full OpenAI API compatibility (v1/chat/completions, v1/completions, v1/embeddings, v1/models)
- Real-time streaming support (SSE)
- Tool calling with proper parallel execution and accumulation
- CORS enabled for browser-based applications
- SQLite logging for all requests and responses
- Token usage tracking
- Admin endpoints: /gateway/status, /gateway/stats, /gateway/logs
- Cross-platform support (Linux, macOS, Windows)
- Single-file portable version (nvidia-ai-gateway.py)
- Python package distribution (pyproject.toml)
- Installation scripts for all platforms (install.sh, install.bat)
- Run scripts (run.sh, run.bat)
- PyInstaller configuration for standalone executables
- GitHub Actions CI/CD workflow for automated builds
- Comprehensive documentation (README.md, INSTALL.md)
- Apache 2.0 license

### Fixed
- **Critical**: Fixed virtual environment creation in restricted environments (ensurepip error)
  - install.sh now uses `--without-pip` flag and manually installs pip via get-pip.py
  - Handles environments where ensurepip is unavailable or fails
  - Detects and repairs incomplete virtual environments
  - Multiple fallback methods (venv, virtualenv package)

- Fixed Flask after_request ordering issue in CORS setup
- Fixed tool call accumulation bug - properly aggregates all tool calls in responses
- Fixed missing imports in terminal client script
- Fixed OpenAI streaming compatibility with proper delta parsing

### Security
- API key validation on startup
- Environment variable-based configuration
- No hardcoded credentials

## [Unreleased]

### Planned
- Rate limiting middleware
- Prometheus metrics endpoint
- WebSocket support for bidirectional streaming
- Docker containerization
- PyPI publication
- Additional authentication options
- Request/response transformation hooks
- Multiple backend support (OpenAI, Anthropic, etc.)
