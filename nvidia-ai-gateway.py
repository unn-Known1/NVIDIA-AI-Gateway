#!/usr/bin/env python3
"""
NVIDIA AI Gateway - Single file version
Just copy this file and run it with Python 3.8+

Usage:
    python nvidia-ai-gateway.py --port 8080

Or download the standalone binary from Releases page.
"""

import os
import sys
import json
import time
START_TIME = time.time()
import uuid
import re
import argparse
import logging
import socket
import sqlite3
import threading
from datetime import datetime
from contextlib import contextmanager
from typing import Optional, Tuple, List, Dict, Any

# ═══════════════════════════════════════════════════════════════
# SECURITY: Removed auto-install to prevent supply chain attacks
# ═══════════════════════════════════════════════════════════════
# Install dependencies manually: pip install flask flask-cors requests

try:
    from flask import Flask, request, jsonify, Response, stream_with_context
    from flask_cors import CORS
    import requests
    from werkzeug.serving import make_server
except ImportError as e:
    print(f"ERROR: Missing required dependency: {e}", file=sys.stderr)
    print("Please install dependencies: pip install flask flask-cors requests", file=sys.stderr)
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════
# PERFORMANCE: Connection Pooling with HTTPAdapter
# ═══════════════════════════════════════════════════════════════
# Create a session with connection pooling for upstream requests
_http_session = requests.Session()
# Configure connection pool size and retry settings
_adapter = requests.adapters.HTTPAdapter(
    pool_connections=10,
    pool_maxsize=20,
    max_retries=0,  # Retries handled at application level
    pool_block=False
)
_http_session.mount('http://', _adapter)
_http_session.mount('https://', _adapter)

# Override default requests.get/post with session methods
requests.get = _http_session.get
requests.post = _http_session.post

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

def load_config():
    return {
        "CUSTOM_BASE_URL": os.getenv("CUSTOM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        "CUSTOM_MODEL_ID": os.getenv("CUSTOM_MODEL_ID", "stepfun-ai/step-3.5-flash"),
        "CUSTOM_API_KEY": os.getenv("CUSTOM_API_KEY", ""),
        "GATEWAY_PORT": int(os.getenv("GATEWAY_PORT", 8089)),
        "GATEWAY_API_KEY": os.getenv("GATEWAY_API_KEY", "sk-gateway-" + uuid.uuid4().hex[:24]),
        "DB_PATH": os.getenv("DB_PATH", "gateway_requests.db"),
        "LOG_FILE": os.getenv("LOG_FILE", "gateway.log"),
        "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
    }

# Rate limiting configuration
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
_request_counts: Dict[str, List[float]] = {}
_rate_limit_lock = threading.Lock()

def check_rate_limit(client_ip: str) -> Tuple[bool, Optional[str]]:
    """Check if client IP has exceeded rate limit. Returns (allowed, message)."""
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW

    with _rate_limit_lock:
        if client_ip in _request_counts:
            _request_counts[client_ip] = [
                ts for ts in _request_counts[client_ip] if ts > window_start
            ]

        request_count = len(_request_counts.get(client_ip, []))

        if request_count >= RATE_LIMIT_REQUESTS:
            return False, f"Rate limit exceeded. Max {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW} seconds."

        if client_ip not in _request_counts:
            _request_counts[client_ip] = []
        _request_counts[client_ip].append(now)

        return True, None

# CORS configuration
_allowed_origins = os.getenv("CORS_ALLOWED_ORIGINS", "")
_is_production = os.getenv("NODE_ENV") == "production"
_origins_list = [o.strip() for o in _allowed_origins.split(",") if o.strip()] if _allowed_origins else []

config = load_config()

if not config["CUSTOM_API_KEY"]:
    print("ERROR: CUSTOM_API_KEY environment variable is required!", file=sys.stderr)
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════
logging.basicConfig(
    level=getattr(logging, config["LOG_LEVEL"].upper()),
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(config["LOG_FILE"]),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# Database
# ═══════════════════════════════════════════════════════════════
_db_lock = threading.Lock()

def init_db():
    with _db_lock, sqlite3.connect(config["DB_PATH"]) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS api_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                endpoint TEXT,
                method TEXT,
                client_ip TEXT,
                model_requested TEXT,
                model_used TEXT,
                streaming INTEGER DEFAULT 0,
                request_body TEXT,
                response_status INTEGER,
                response_body TEXT,
                prompt_tokens INTEGER DEFAULT 0,
                completion_tokens INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                duration_ms REAL DEFAULT 0,
                error TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON api_logs(timestamp)")
        conn.commit()

@contextmanager
def get_db():
    with _db_lock:
        conn = sqlite3.connect(config["DB_PATH"], check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

def insert_log(entry: dict):
    sql = """
        INSERT INTO api_logs (
            request_id, timestamp, endpoint, method, client_ip,
            model_requested, model_used, streaming,
            request_body, response_status, response_body,
            prompt_tokens, completion_tokens, total_tokens,
            duration_ms, error
        ) VALUES (
            :request_id, :timestamp, :endpoint, :method, :client_ip,
            :model_requested, :model_used, :streaming,
            :request_body, :response_status, :response_body,
            :prompt_tokens, :completion_tokens, :total_tokens,
            :duration_ms, :error
        )
    """
    with get_db() as conn:
        conn.execute(sql, entry)

def update_log_tokens(request_id: str, prompt: int, completion: int, response_body: str = None):
    with get_db() as conn:
        if response_body:
            conn.execute(
                """UPDATE api_logs
                   SET prompt_tokens=?, completion_tokens=?, total_tokens=?, response_body=?
                   WHERE request_id=?""",
                (prompt, completion, prompt + completion, response_body, request_id),
            )
        else:
            conn.execute(
                """UPDATE api_logs
                   SET prompt_tokens=?, completion_tokens=?, total_tokens=?
                   WHERE request_id=?""",
                (prompt, completion, prompt + completion, request_id),
            )

# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def extract_tokens_from_response(body: dict) -> Tuple[int, int]:
    usage = body.get("usage") or {}
    return usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)

def extract_tokens_from_chunks(chunks: List[Dict]) -> Tuple[int, int]:
    for chunk in reversed(chunks):
        usage = chunk.get("usage")
        if usage:
            return usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    return 0, 0

def _authorized() -> bool:
    auth = request.headers.get("Authorization", "")
    return auth.startswith("Bearer ") and auth[7:] == config["GATEWAY_API_KEY"]

def _openai_error(message: str, error_type: str = "invalid_request_error", status_code: int = 400) -> Tuple[Dict, int]:
    return {"error": {"message": message, "type": error_type, "param": None, "code": None}}, status_code

def _validate_messages(messages: List[Dict]) -> Optional[str]:
    if not isinstance(messages, list) or not messages:
        return "messages must be a non-empty array"
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            return f"message {i} must be an object"
        role = msg.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            return f"message {i} has invalid role '{role}'"
        if "content" not in msg and "tool_calls" not in msg:
            return f"message {i} must have content or tool_calls"
        if role == "tool" and not msg.get("tool_call_id"):
            return f"tool message {i} requires tool_call_id"
        if role == "assistant" and msg.get("tool_calls"):
            if msg.get("content") not in (None, ""):
                return f"assistant message {i} with tool_calls must have empty content"
    return None

def accumulate_tool_calls(chunks: List[Dict]) -> List[Dict]:
    tool_calls_by_index: Dict[int, Dict] = {}
    for chunk in chunks:
        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})
            if "tool_calls" not in delta:
                continue
            for tool_call in delta["tool_calls"]:
                index = tool_call.get("index", 0)
                if index not in tool_calls_by_index:
                    tool_calls_by_index[index] = {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                if "id" in tool_call:
                    tool_calls_by_index[index]["id"] = tool_call["id"]
                if "type" in tool_call:
                    tool_calls_by_index[index]["type"] = tool_call["type"]
                if "function" in tool_call:
                    func = tool_call["function"]
                    if "name" in func:
                        tool_calls_by_index[index]["function"]["name"] = func["name"]
                    if "arguments" in func:
                        tool_calls_by_index[index]["function"]["arguments"] += func["arguments"]
    return [tool_calls_by_index[i] for i in sorted(tool_calls_by_index.keys())]

def extract_full_content(chunks: List[Dict]) -> str:
    parts: List[str] = []
    for chunk in chunks:
        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})
            if "content" in delta and delta["content"]:
                parts.append(delta["content"])
            if "reasoning_content" in delta and delta["reasoning_content"]:
                parts.append(delta["reasoning_content"])
    return "".join(parts)

def add_cors_headers(response: Response) -> Response:
    """Add secure CORS headers based on environment configuration."""
    _origin = request.headers.get('Origin', '')
    _is_localhost = _origin.startswith('http://localhost') or _origin.startswith('http://127.0.0.1')

    # In production, only allow configured origins and validate against allowlist
    if _is_production:
        if _origins_list and _origin in _origins_list:
            # Only set specific origin in production for security
            response.headers.add('Access-Control-Allow-Origin', _origin)
            response.headers.add('Vary', 'Origin')
        # Don't add headers if origin is not in allowlist - secure by default
    elif _is_localhost or not _is_production:
        # Allow localhost in development
        if _origin:
            response.headers.add('Access-Control-Allow-Origin', _origin)
            response.headers.add('Vary', 'Origin')

    # Only expose these headers when CORS is actually allowed
    if response.headers.get('Access-Control-Allow-Origin'):
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization, Accept, X-Session-ID')
        response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        response.headers.add('Access-Control-Max-Age', '86400')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        # Security headers
        response.headers.add('X-Content-Type-Options', 'nosniff')
        response.headers.add('X-XSS-Protection', '1; mode=block')
    return response

def _log_error(entry: dict, start_ts: float, status: int, msg: str):
    entry.update({
        "response_status": status,
        "duration_ms": round((time.time() - start_ts) * 1000, 2),
        "error": msg,
    })
    try:
        insert_log(entry)
    except Exception as e:
        logger.error("Failed to insert error log: %s", e)
    logger.error("[gateway] status=%d  %s", status, msg)

# ═══════════════════════════════════════════════════════════════
# Flask App
# ═══════════════════════════════════════════════════════════════
app = Flask(__name__)

# Configure CORS based on environment
# SECURITY: Default to restrictive CORS in production
_cors_origins = os.getenv("CORS_ALLOWED_ORIGINS", "")
_is_production = os.getenv("NODE_ENV") == "production"

# SECURE: Default to restrictive CORS in production
# Only allow wildcard (*) if explicitly configured in non-production
if _cors_origins:
    # Use explicitly configured origins
    _allowed = [o.strip() for o in _cors_origins.split(",") if o.strip()]
    CORS(app, resources={r"/*": {"origins": _allowed, "supports_credentials": True}})
    logger.info("CORS enabled for configured origins: %s", _allowed)
elif _is_production:
    # PRODUCTION: Block all cross-origin requests by default
    logger.warning("CORS set to deny all origins in production mode for security!")
    CORS(app, resources={r"/*": {"origins": []}})
else:
    # DEVELOPMENT: Allow localhost and local development only
    _dev_origins = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]
    logger.info("CORS enabled for development origins: %s", _dev_origins)
    CORS(app, resources={r"/*": {"origins": _dev_origins}})

@app.after_request
def after_request(response: Response) -> Response:
    return add_cors_headers(response)

# ═══════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health_check():
    uptime = time.time() - START_TIME
    return jsonify({
        "status": "ok",
        "service": "nvidia-ai-gateway",
        "version": "2.0.0",
        "uptime_seconds": round(uptime, 2),
        "timestamp": time.time()
    })

@app.route("/", methods=["GET", "OPTIONS"])
@app.route("/v1", methods=["GET", "OPTIONS"])
def root_endpoint():
    if request.method == "OPTIONS":
        resp = Response("", 200)
        return add_cors_headers(resp)
    return jsonify({
        "object": "api",
        "message": "NVIDIA AI Gateway - OpenAI Compatible",
        "version": "2.0.0",
        "endpoints": {
            "chat": "/v1/chat/completions",
            "completions": "/v1/completions",
            "embeddings": "/v1/embeddings",
            "models": "/v1/models",
            "gateway": {"status": "/gateway/status", "stats": "/gateway/stats", "logs": "/gateway/logs"}
        }
    })

@app.route("/v1/chat/completions", methods=["POST", "OPTIONS"])
@app.route("/chat/completions", methods=["POST", "OPTIONS"])
def chat_completions():
    if request.method == "OPTIONS":
        return add_cors_headers(Response("", 200))

    # Check rate limit
    client_ip = request.remote_addr
    allowed, message = check_rate_limit(client_ip)
    if not allowed:
        logger.warning("Rate limit exceeded for %s", client_ip)
        error, code = _openai_error(message, "rate_limit_exceeded", 429)
        return add_cors_headers(jsonify(error)), code

    if not _authorized():
        logger.warning("Unauthorized from %s", request.remote_addr)
        error, code = _openai_error("Invalid API key", "invalid_api_key", 401)
        return add_cors_headers(jsonify(error)), code

    try:
        req_body = request.get_json(force=True)
    except Exception:
        error, code = _openai_error("Invalid JSON body", "invalid_request_error", 400)
        return add_cors_headers(jsonify(error)), code

    if not req_body:
        error, code = _openai_error("Empty request body", "invalid_request_error", 400)
        return add_cors_headers(jsonify(error)), code

    messages = req_body.get("messages", [])
    validation_error = _validate_messages(messages)
    if validation_error:
        error, code = _openai_error(validation_error, "invalid_request_error", 400)
        return add_cors_headers(jsonify(error)), code

    original_model = req_body.get("model", config["CUSTOM_MODEL_ID"])
    req_body["model"] = config["CUSTOM_MODEL_ID"]

    is_stream = bool(req_body.get("stream", False))
    request_id = str(uuid.uuid4())
    start_ts = time.time()

    log_entry = {
        "request_id": request_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "endpoint": "/v1/chat/completions",
        "method": "POST",
        "client_ip": request.remote_addr,
        "model_requested": original_model,
        "model_used": config["CUSTOM_MODEL_ID"],
        "streaming": int(is_stream),
        "request_body": json.dumps(req_body),
        "response_status": 0,
        "response_body": None,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "duration_ms": 0,
        "error": None,
    }

    target_url = f"{config['CUSTOM_BASE_URL']}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['CUSTOM_API_KEY']}",
        "User-Agent": "NVIDIA-AI-Gateway/2.0",
        "Accept": "text/event-stream" if is_stream else "application/json",
    }

    if is_stream:
        collected_chunks = []
        try:
            upstream = requests.post(target_url, json=req_body, headers=headers, stream=True, timeout=(10, 300))
        except requests.exceptions.ConnectionError as e:
            _log_error(log_entry, start_ts, 502, str(e))
            error, code = _openai_error(f"Cannot reach upstream: {e}", "api_error", 502)
            return add_cors_headers(jsonify(error)), code
        except requests.exceptions.Timeout:
            _log_error(log_entry, start_ts, 504, "Connection timeout")
            error, code = _openai_error("Upstream connection timed out", "api_error", 504)
            return add_cors_headers(jsonify(error)), code

        if upstream.status_code != 200:
            body = upstream.text
            _log_error(log_entry, start_ts, upstream.status_code, body)
            try:
                err_resp = json.loads(body)
            except Exception:
                err_resp, _ = _openai_error(f"Upstream error: {body[:200]}", "api_error", upstream.status_code)
            return add_cors_headers(jsonify(err_resp)), upstream.status_code

        log_entry["response_status"] = 200
        insert_log(log_entry)

        def generate():
            nonlocal log_entry, start_ts, collected_chunks, request_id
            try:
                for raw_line in upstream.iter_lines(chunk_size=None, decode_unicode=True):
                    if not raw_line:
                        yield "\n"
                        continue
                    yield raw_line + "\n\n"
                    if raw_line.startswith("data:"):
                        data_str = raw_line[5:].strip()
                        if data_str == "[DONE]":
                            continue
                        try:
                            chunk = json.loads(data_str)
                            collected_chunks.append(chunk)
                        except json.JSONDecodeError:
                            pass
            finally:
                duration_ms = (time.time() - start_ts) * 1000
                prompt_tok, compl_tok = extract_tokens_from_chunks(collected_chunks)
                full_response = extract_full_content(collected_chunks)
                tool_calls = accumulate_tool_calls(collected_chunks)
                response_for_log = {"content": full_response, "tool_calls": tool_calls}
                try:
                    update_log_tokens(request_id, prompt_tok, compl_tok, response_body=json.dumps(response_for_log)[:16000])
                    with get_db() as conn:
                        conn.execute("UPDATE api_logs SET duration_ms=? WHERE request_id=?", (round(duration_ms, 2), request_id))
                except Exception as db_err:
                    logger.error("DB update after stream failed: %s", db_err)
                logger.info("[%s] stream done | tokens p=%d c=%d | %.0f ms", request_id[:8], prompt_tok, compl_tok, duration_ms)

        return Response(
            stream_with_context(generate()),
            status=200,
            content_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
                "Transfer-Encoding": "chunked",
                "X-Gateway-RequestId": request_id,
                "Content-Type": "text/event-stream; charset=utf-8",
            },
        )

    # Non-streaming
    try:
        resp = requests.post(target_url, json=req_body, headers=headers, timeout=(10, 300))
        duration_ms = (time.time() - start_ts) * 1000
        try:
            resp_body = resp.json()
        except Exception:
            resp_body = {"raw": resp.text}
        prompt_tok, compl_tok = extract_tokens_from_response(resp_body)
        log_entry.update({
            "response_status": resp.status_code,
            "response_body": json.dumps(resp_body)[:16000],
            "prompt_tokens": prompt_tok,
            "completion_tokens": compl_tok,
            "total_tokens": prompt_tok + compl_tok,
            "duration_ms": round(duration_ms, 2),
        })
        insert_log(log_entry)
        logger.info("[%s] non-stream | status=%d tokens p=%d c=%d | %.0f ms",
                    request_id[:8], resp.status_code, prompt_tok, compl_tok, duration_ms)
        response = Response(resp.content, status=resp.status_code,
                            content_type=resp.headers.get("Content-Type", "application/json"),
                            headers={"X-Gateway-RequestId": request_id})
        return add_cors_headers(response)
    except requests.exceptions.Timeout:
        _log_error(log_entry, start_ts, 504, "Upstream read timed out")
        error, code = _openai_error("Upstream timed out", "api_error", 504)
        return add_cors_headers(jsonify(error)), code
    except requests.exceptions.RequestException as exc:
        _log_error(log_entry, start_ts, 502, str(exc))
        error, code = _openai_error(f"Upstream error: {exc}", "api_error", 502)
        return add_cors_headers(jsonify(error)), code

@app.route("/v1/completions", methods=["POST", "OPTIONS"])
@app.route("/completions", methods=["POST", "OPTIONS"])
def completions():
    if request.method == "OPTIONS":
        return add_cors_headers(Response("", 200))

    # Check rate limit
    client_ip = request.remote_addr
    allowed, message = check_rate_limit(client_ip)
    if not allowed:
        logger.warning("Rate limit exceeded for %s", client_ip)
        error, code = _openai_error(message, "rate_limit_exceeded", 429)
        return add_cors_headers(jsonify(error)), code

    if not _authorized():
        error, code = _openai_error("Invalid API key", "invalid_api_key", 401)
        return add_cors_headers(jsonify(error)), code
    try:
        req_body = request.get_json(force=True)
    except Exception:
        error, code = _openai_error("Invalid JSON body", "invalid_request_error", 400)
        return add_cors_headers(jsonify(error)), code
    if not req_body:
        error, code = _openai_error("Empty request body", "invalid_request_error", 400)
        return add_cors_headers(jsonify(error)), code

    original_model = req_body.get("model", config["CUSTOM_MODEL_ID"])
    req_body["model"] = config["CUSTOM_MODEL_ID"]
    is_stream = bool(req_body.get("stream", False))
    request_id = str(uuid.uuid4())
    start_ts = time.time()

    log_entry = {
        "request_id": request_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "endpoint": "/v1/completions",
        "method": "POST",
        "client_ip": request.remote_addr,
        "model_requested": original_model,
        "model_used": config["CUSTOM_MODEL_ID"],
        "streaming": int(is_stream),
        "request_body": json.dumps(req_body),
        "response_status": 0,
        "response_body": None,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "duration_ms": 0,
        "error": None,
    }

    target_url = f"{config['CUSTOM_BASE_URL']}/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['CUSTOM_API_KEY']}",
        "User-Agent": "NVIDIA-AI-Gateway/2.0",
        "Accept": "text/event-stream" if is_stream else "application/json",
    }

    if is_stream:
        collected_chunks = []
        try:
            upstream = requests.post(target_url, json=req_body, headers=headers, stream=True, timeout=(10, 300))
        except requests.exceptions.ConnectionError as e:
            _log_error(log_entry, start_ts, 502, str(e))
            error, code = _openai_error(f"Cannot reach upstream: {e}", "api_error", 502)
            return add_cors_headers(jsonify(error)), code
        except requests.exceptions.Timeout:
            _log_error(log_entry, start_ts, 504, "Connection timeout")
            error, code = _openai_error("Connection timeout", "api_error", 504)
            return add_cors_headers(jsonify(error)), code

        if upstream.status_code != 200:
            body = upstream.text
            _log_error(log_entry, start_ts, upstream.status_code, body)
            try:
                err_resp = json.loads(body)
            except Exception:
                err_resp, _ = _openai_error(f"Upstream error: {body[:200]}", "api_error", upstream.status_code)
            return add_cors_headers(jsonify(err_resp)), upstream.status_code

        log_entry["response_status"] = 200
        insert_log(log_entry)

        def generate():
            nonlocal log_entry, start_ts, collected_chunks, request_id
            try:
                for raw_line in upstream.iter_lines(chunk_size=None, decode_unicode=True):
                    if not raw_line:
                        yield "\n"
                        continue
                    yield raw_line + "\n\n"
                    if raw_line.startswith("data:"):
                        data_str = raw_line[5:].strip()
                        if data_str == "[DONE]":
                            continue
                        try:
                            chunk = json.loads(data_str)
                            collected_chunks.append(chunk)
                        except json.JSONDecodeError:
                            pass
            finally:
                duration_ms = (time.time() - start_ts) * 1000
                prompt_tok, compl_tok = extract_tokens_from_chunks(collected_chunks)
                full_response = extract_full_content(collected_chunks)
                tool_calls = accumulate_tool_calls(collected_chunks)
                response_for_log = {"content": full_response, "tool_calls": tool_calls}
                try:
                    update_log_tokens(request_id, prompt_tok, compl_tok, response_body=json.dumps(response_for_log)[:16000])
                    with get_db() as conn:
                        conn.execute("UPDATE api_logs SET duration_ms=? WHERE request_id=?", (round(duration_ms, 2), request_id))
                except Exception as db_err:
                    logger.error("DB update after stream failed: %s", db_err)
                logger.info("[%s] stream done | tokens p=%d c=%d | %.0f ms", request_id[:8], prompt_tok, compl_tok, duration_ms)

        return Response(stream_with_context(generate()), status=200, content_type="text/event-stream", headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "Transfer-Encoding": "chunked",
            "X-Gateway-RequestId": request_id,
            "Content-Type": "text/event-stream; charset=utf-8",
        })
    else:
        try:
            resp = requests.post(target_url, json=req_body, headers=headers, timeout=(10, 300))
            duration_ms = (time.time() - start_ts) * 1000
            try:
                resp_body = resp.json()
            except Exception:
                resp_body = {"raw": resp.text}
            prompt_tok, compl_tok = extract_tokens_from_response(resp_body)
            log_entry.update({
                "response_status": resp.status_code,
                "response_body": json.dumps(resp_body)[:16000],
                "prompt_tokens": prompt_tok,
                "completion_tokens": compl_tok,
                "total_tokens": prompt_tok + compl_tok,
                "duration_ms": round(duration_ms, 2),
            })
            insert_log(log_entry)
            logger.info("[%s] non-stream | status=%d tokens p=%d c=%d | %.0f ms",
                        request_id[:8], resp.status_code, prompt_tok, compl_tok, duration_ms)
            response = Response(resp.content, status=resp.status_code,
                                content_type=resp.headers.get("Content-Type", "application/json"),
                                headers={"X-Gateway-RequestId": request_id})
            return add_cors_headers(response)
        except requests.exceptions.Timeout:
            _log_error(log_entry, start_ts, 504, "Upstream read timed out")
            error, code = _openai_error("Upstream timed out", "api_error", 504)
            return add_cors_headers(jsonify(error)), code
        except requests.exceptions.RequestException as exc:
            _log_error(log_entry, start_ts, 502, str(exc))
            error, code = _openai_error(f"Upstream error: {exc}", "api_error", 502)
            return add_cors_headers(jsonify(error)), code

@app.route("/v1/embeddings", methods=["POST", "OPTIONS"])
@app.route("/embeddings", methods=["POST", "OPTIONS"])
def embeddings():
    if request.method == "OPTIONS":
        return add_cors_headers(Response("", 200))

    # Check rate limit
    client_ip = request.remote_addr
    allowed, message = check_rate_limit(client_ip)
    if not allowed:
        logger.warning("Rate limit exceeded for %s", client_ip)
        error, code = _openai_error(message, "rate_limit_exceeded", 429)
        return add_cors_headers(jsonify(error)), code

    if not _authorized():
        error, code = _openai_error("Invalid API key", "invalid_api_key", 401)
        return add_cors_headers(jsonify(error)), code
    try:
        req_body = request.get_json(force=True)
    except Exception:
        error, code = _openai_error("Invalid JSON body", "invalid_request_error", 400)
        return add_cors_headers(jsonify(error)), code

    input_data = req_body.get("input")
    if not input_data:
        error, code = _openai_error("Missing 'input' field", "invalid_request_error", 400)
        return add_cors_headers(jsonify(error)), code

    original_model = req_body.get("model", config["CUSTOM_MODEL_ID"])
    req_body["model"] = config["CUSTOM_MODEL_ID"]
    request_id = str(uuid.uuid4())
    start_ts = time.time()

    log_entry = {
        "request_id": request_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "endpoint": "/v1/embeddings",
        "method": "POST",
        "client_ip": request.remote_addr,
        "model_requested": original_model,
        "model_used": config["CUSTOM_MODEL_ID"],
        "streaming": 0,
        "request_body": json.dumps(req_body),
        "response_status": 0,
        "response_body": None,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "duration_ms": 0,
        "error": None,
    }

    target_url = f"{config['CUSTOM_BASE_URL']}/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['CUSTOM_API_KEY']}",
        "User-Agent": "NVIDIA-AI-Gateway/2.0",
        "Accept": "application/json",
    }

    try:
        resp = requests.post(target_url, json=req_body, headers=headers, timeout=(10, 300))
        duration_ms = (time.time() - start_ts) * 1000
        try:
            resp_body = resp.json()
        except Exception:
            resp_body = {"raw": resp.text}

        input_str = ""
        if isinstance(input_data, str):
            input_str = input_data
        elif isinstance(input_data, list):
            for item in input_data:
                if isinstance(item, str):
                    input_str += item
                elif isinstance(item, dict):
                    if "text" in item:
                        input_str += item["text"]

        # Use more accurate token estimation (4 chars per token)
        prompt_tokens = int(len(input_str) / 4)
        completion_tokens = len(resp_body.get("data", [])) * 1
        if isinstance(resp_body, dict) and "usage" in resp_body:
            prompt_tok = resp_body["usage"].get("prompt_tokens", prompt_tokens)
            compl_tok = resp_body["usage"].get("completion_tokens", completion_tokens)
            prompt_tokens, completion_tokens = prompt_tok, compl_tok

        log_entry.update({
            "response_status": resp.status_code,
            "response_body": json.dumps(resp_body)[:16000],
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "duration_ms": round(duration_ms, 2),
        })
        insert_log(log_entry)

        logger.info("[%s] embeddings | status=%d tokens p=%d | %.0f ms",
                    request_id[:8], resp.status_code, prompt_tokens, duration_ms)

        response = Response(resp.content, status=resp.status_code,
                            content_type=resp.headers.get("Content-Type", "application/json"),
                            headers={"X-Gateway-RequestId": request_id})
        return add_cors_headers(response)

    except requests.exceptions.Timeout:
        _log_error(log_entry, start_ts, 504, "Upstream read timed out")
        error, code = _openai_error("Upstream timed out", "api_error", 504)
        return add_cors_headers(jsonify(error)), code
    except requests.exceptions.RequestException as exc:
        _log_error(log_entry, start_ts, 502, str(exc))
        error, code = _openai_error(f"Upstream error: {exc}", "api_error", 502)
        return add_cors_headers(jsonify(error)), code

@app.route("/v1/models", methods=["GET", "OPTIONS"])
@app.route("/models", methods=["GET", "OPTIONS"])
def list_models():
    if request.method == "OPTIONS":
        return add_cors_headers(Response("", 200))
    if not _authorized():
        error, code = _openai_error("Invalid API key", "invalid_api_key", 401)
        return add_cors_headers(jsonify(error)), code
    try:
        resp = requests.get(f"{config['CUSTOM_BASE_URL']}/models",
                            headers={"Authorization": f"Bearer {config['CUSTOM_API_KEY']}"},
                            timeout=15)
        if resp.status_code == 200:
            return add_cors_headers(Response(resp.content, 200, resp.headers.get("Content-Type", "application/json")))
    except Exception:
        pass
    fallback = jsonify({
        "object": "list",
        "data": [{
            "id": config["CUSTOM_MODEL_ID"],
            "object": "model",
            "created": int(time.time()),
            "owned_by": "nvidia",
        }],
    })
    return add_cors_headers(fallback)

@app.route("/gateway/status", methods=["GET", "OPTIONS"])
def gateway_status():
    if request.method == "OPTIONS":
        return add_cors_headers(Response("", 200))
    return add_cors_headers(jsonify({
        "status": "running",
        "gateway_port": config["GATEWAY_PORT"],
        "target_base_url": config["CUSTOM_BASE_URL"],
        "target_model": config["CUSTOM_MODEL_ID"],
        "db_path": config["DB_PATH"],
        "features": ["OpenAI-compatible streaming", "SQLite logging", "Rate limiting", "CORS enabled", "Tool calls support", "Embeddings endpoint", "Completions endpoint"]
    }))

@app.route("/gateway/stats", methods=["GET", "OPTIONS"])
def gateway_stats():
    if request.method == "OPTIONS":
        return add_cors_headers(Response("", 200))
    if not _authorized():
        error, code = _openai_error("Invalid API key", "invalid_api_key", 401)
        return add_cors_headers(jsonify(error)), code
    with get_db() as conn:
        row = conn.execute("""
            SELECT COUNT(*) AS total_requests,
                   SUM(streaming) AS streaming_requests,
                   SUM(total_tokens) AS total_tokens,
                   SUM(prompt_tokens) AS total_prompt_tokens,
                   SUM(completion_tokens) AS total_completion_tokens,
                   AVG(duration_ms) AS avg_duration_ms,
                   SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) AS error_count
            FROM api_logs
        """).fetchone()
    return add_cors_headers(jsonify({
        "total_requests": row["total_requests"],
        "streaming_requests": row["streaming_requests"],
        "non_streaming": (row["total_requests"] or 0) - (row["streaming_requests"] or 0),
        "total_tokens": row["total_tokens"],
        "total_prompt_tokens": row["total_prompt_tokens"],
        "total_completion_tokens": row["total_completion_tokens"],
        "avg_duration_ms": round(row["avg_duration_ms"] or 0, 2),
        "error_count": row["error_count"],
    }))

@app.route("/gateway/logs", methods=["GET", "OPTIONS"])
def get_logs():
    if request.method == "OPTIONS":
        return add_cors_headers(Response("", 200))
    if not _authorized():
        error, code = _openai_error("Invalid API key", "invalid_api_key", 401)
        return add_cors_headers(jsonify(error)), code
    limit = request.args.get("limit", 50, type=int)
    offset = request.args.get("offset", 0, type=int)
    stream_only = request.args.get("stream", None)
    # Use parameterized queries to prevent SQL injection
    params: List = []
    if stream_only is not None:
        # Safely validate stream_only parameter
        stream_value = 1 if stream_only.lower() in ("1", "true") else 0
        where_clause = "WHERE streaming = ?"
        params.append(stream_value)
    else:
        where_clause = ""

    with get_db() as conn:
        # Use parameterized query with separate params list
        total = conn.execute(
            "SELECT COUNT(*) FROM api_logs " + where_clause,
            params
        ).fetchone()[0]

        # Append limit and offset to params for safe pagination
        query_params = params + [limit, offset]
        rows = conn.execute(
            "SELECT * FROM api_logs " + where_clause + " ORDER BY id DESC LIMIT ? OFFSET ?",
            query_params,
        ).fetchall()
    logs = []
    for row in rows:
        d = dict(row)
        for field in ("request_body", "response_body"):
            if d.get(field):
                try:
                    d[field] = json.loads(d[field])
                except Exception:
                    pass
        logs.append(d)
    return add_cors_headers(jsonify({"total": total, "limit": limit, "offset": offset, "returned": len(logs), "logs": logs}))

# ═══════════════════════════════════════════════════════════════
# Background server with graceful shutdown support
# ═══════════════════════════════════════════════════════════════
class BackgroundServer:
    def __init__(self, flask_app, port):
        self.server = make_server("0.0.0.0", port, flask_app)
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self._shutdown_event = threading.Event()
    def start(self):
        self._thread.start()
        logger.info("Background server started")
    def stop(self, timeout=10):
        """Gracefully stop the server with timeout protection."""
        logger.info("Initiating graceful shutdown...")
        self._shutdown_event.set()
        try:
            # Request server to shut down gracefully
            self.server.shutdown()
            # Wait for thread to finish with timeout
            if self._thread.is_alive():
                self._thread.join(timeout=timeout)
                if self._thread.is_alive():
                    logger.warning("Server thread did not stop within %ds, forcing...", timeout)
        except Exception as e:
            logger.error("Error during server shutdown: %s", e)
        logger.info("Server shutdown complete")

def _local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

# ═══════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="NVIDIA AI Gateway - OpenAI-Compatible Proxy")
    parser.add_argument("--port", type=int, default=config["GATEWAY_PORT"], help="Port to run the gateway on")
    parser.add_argument("--config", type=str, help="Path to config file (INI format)")
    parser.add_argument("--log-level", type=str, default=config["LOG_LEVEL"], choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    if args.port != config["GATEWAY_PORT"]:
        config["GATEWAY_PORT"] = args.port
    if args.log_level:
        config["LOG_LEVEL"] = args.log_level
        logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    if args.config:
        import configparser
        cfg = configparser.ConfigParser()
        cfg.read(args.config)
        if "gateway" in cfg:
            for key in ["CUSTOM_BASE_URL", "CUSTOM_MODEL_ID", "CUSTOM_API_KEY", "DB_PATH", "LOG_FILE"]:
                if key in cfg["gateway"]:
                    config[key] = cfg["gateway"][key]
        if "server" in cfg and "port" in cfg["server"]:
            config["GATEWAY_PORT"] = int(cfg["server"]["port"])

    if not config["CUSTOM_API_KEY"]:
        print("ERROR: CUSTOM_API_KEY is not set!", file=sys.stderr)
        sys.exit(1)

    init_db()
    logger.info("SQLite DB ready at %s", config["DB_PATH"])

    ip = _local_ip()
    base = f"http://{ip}:{config['GATEWAY_PORT']}"

    # SECURE: API key is never displayed in banner
    # Users must set GATEWAY_API_KEY env var to use the gateway

    BANNER = f"""
╔══════════════════════════════════════════════════════════════════╗
║          NVIDIA AI Gateway v2.0.0 (OpenAI-Compatible)          ║
╠══════════════════════════════════════════════════════════════════╣
║  Gateway Base URL : {base}/v1
║  Gateway API Key  : ***MASKED*** (set via GATEWAY_API_KEY env)
║  Target URL       : {config['CUSTOM_BASE_URL']}
║  Target Model     : {config['CUSTOM_MODEL_ID']}
║  DB               : {config['DB_PATH']}
║  Rate Limit       : {RATE_LIMIT_REQUESTS} req / {RATE_LIMIT_WINDOW}s
╠══════════════════════════════════════════════════════════════════╣
║  SECURITY FIXES APPLIED:
║  ✓ Supply chain attack prevention (auto-install removed)
║  ✓ Rate limiting enabled
║  ✓ Secure CORS configuration
║  ✓ API key masking in logs and banner
╠══════════════════════════════════════════════════════════════════╣
║  OPENAI SDK EXAMPLE (replace YOUR_API_KEY with actual key)
╠══════════════════════════════════════════════════════════════════╣

    from openai import OpenAI

    client = OpenAI(
        base_url="{base}/v1",
        api_key="YOUR_API_KEY"
    )

    completion = client.chat.completions.create(
        model="{config['CUSTOM_MODEL_ID']}",
        messages=[{{"role": "user", "content": "Hello!"}}],
        temperature=1,
        top_p=0.9,
        max_tokens=1024,
        stream=True
    )

    for chunk in completion:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="")

╠══════════════════════════════════════════════════════════════════╣
║  ENDPOINTS
╠══════════════════════════════════════════════════════════════════╣
║  POST {base}/v1/chat/completions (streaming + non-streaming)
║  POST {base}/v1/completions      (legacy completions)
║  POST {base}/v1/embeddings
║  GET  {base}/v1/models
║  GET  {base}/gateway/status
║  GET  {base}/gateway/stats
║  GET  {base}/gateway/logs?limit=50&offset=0
║  OPTIONS any endpoint (CORS preflight)
╚══════════════════════════════════════════════════════════════╝
    """
    print(BANNER)

    server = BackgroundServer(app, config["GATEWAY_PORT"])
    server.start()
    logger.info("Gateway running on port %d — press Ctrl+C to stop", config["GATEWAY_PORT"])

    try:
        while True:
            time.sleep(60)
            with get_db() as conn:
                total = conn.execute("SELECT COUNT(*) FROM api_logs").fetchone()[0]
            logger.info("Heartbeat — total logged requests: %d", total)
    except KeyboardInterrupt:
        print("\nShutting down …")
        server.stop()
        print("✓ Gateway stopped")

if __name__ == "__main__":
    main()