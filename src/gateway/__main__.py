#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
NVIDIA AI Gateway - Production-Ready OpenAI-Compatible Proxy
Supports OpenAI-compatible clients, SSE streaming, tool calls, CORS, SQLite logging.
Cross-platform: Linux, Windows, macOS.

Usage:
    python -m gateway [--port PORT] [--config CONFIG_FILE]
    nvidia-ai-gateway --port 8080 --config config.ini

Configuration via environment variables or config file.
"""

import os
import sys
import json
import time
import uuid
import hmac
import argparse
import logging
import socket
import sqlite3
import threading
import signal
import re
from datetime import datetime, timezone, timedelta
from contextlib import contextmanager
from typing import Optional, Tuple, List, Dict, Any

from dotenv import load_dotenv
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from flask_compress import Compress
import requests
from werkzeug.serving import make_server

# Load environment variables from .env file (IMP-030)
load_dotenv()

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

def load_config():
    """Load configuration from environment variables or defaults."""
    return {
        "CUSTOM_BASE_URL": os.getenv("CUSTOM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        "CUSTOM_MODEL_ID": os.getenv("CUSTOM_MODEL_ID", "stepfun-ai/step-3.5-flash"),
        "CUSTOM_API_KEY": os.getenv("CUSTOM_API_KEY", ""),
        "GATEWAY_HOST": os.getenv("GATEWAY_HOST", "127.0.0.1"),
        "GATEWAY_PORT": int(os.getenv("GATEWAY_PORT", 8089)),
        "GATEWAY_API_KEY": os.getenv("GATEWAY_API_KEY", "sk-gateway-" + uuid.uuid4().hex[:24]),
        "GATEWAY_CORS_ORIGINS": os.getenv("GATEWAY_CORS_ORIGINS", "http://localhost,http://127.0.0.1"),
        "LOG_REQUEST_BODIES": os.getenv("LOG_REQUEST_BODIES", "true").lower() in ("1", "true", "yes"),
        "DB_PATH": os.getenv("DB_PATH", "gateway_requests.db"),
        "LOG_FILE": os.getenv("LOG_FILE", "gateway.log"),
        "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
        "UPSTREAM_CONNECT_TIMEOUT": int(os.getenv("UPSTREAM_CONNECT_TIMEOUT", "10")),
        "UPSTREAM_READ_TIMEOUT": int(os.getenv("UPSTREAM_READ_TIMEOUT", "300")),
        "ALLOWED_MODELS": os.getenv("ALLOWED_MODELS", "*"),
        "LOG_RETENTION_DAYS": int(os.getenv("LOG_RETENTION_DAYS", "30")),
        "UPSTREAM_RETRIES": int(os.getenv("UPSTREAM_RETRIES", "3")),
    }

config = load_config()

# NOTE: CUSTOM_API_KEY validation is deferred to main() so that --config
# can supply it (BUG-018). Do NOT sys.exit here.

# Create session for connection pooling (IMP-036)
upstream_session = requests.Session()

# ═══════════════════════════════════════════════════════════════
# Logging setup
# ═══════════════════════════════════════════════════════════════
from pythonjsonlogger import jsonlogger

logging.basicConfig(
    level=getattr(logging, config["LOG_LEVEL"].upper()),
    handlers=[
        logging.FileHandler(config["LOG_FILE"]),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# IMP-011: Structured JSON logging
formatter = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(message)s')
for handler in logging.getLogger().handlers:
    handler.setFormatter(formatter)

# ═══════════════════════════════════════════════════════════════
# Database helpers
# ═══════════════════════════════════════════════════════════════
_db_lock = threading.RLock()

def init_db():
    """Create tables if they don't exist. Enable WAL for concurrent reads."""
    with _db_lock, sqlite3.connect(config["DB_PATH"]) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS api_logs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id      TEXT    NOT NULL,
                timestamp       TEXT    NOT NULL,
                endpoint        TEXT,
                method          TEXT,
                client_ip       TEXT,
                model_requested TEXT,
                model_used      TEXT,
                streaming       INTEGER DEFAULT 0,
                request_body    TEXT,
                response_status INTEGER,
                response_body   TEXT,
                prompt_tokens   INTEGER DEFAULT 0,
                completion_tokens INTEGER DEFAULT 0,
                total_tokens    INTEGER DEFAULT 0,
                duration_ms     REAL    DEFAULT 0,
                error           TEXT
            )
            )
        """)
        # IMP-010: Audit log table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT    NOT NULL,
                event_type      TEXT    NOT NULL,
                details         TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON api_logs(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_request_id ON api_logs(request_id)")
        current_version = conn.execute("PRAGMA user_version").fetchone()[0]
        if current_version < 1:
            conn.execute("PRAGMA user_version = 1")
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

def insert_audit_log(event_type: str, details: dict = None):
    """Insert an audit log event (IMP-010)."""
    with get_db() as conn:
        conn.execute(
            "INSERT INTO audit_logs (timestamp, event_type, details) VALUES (?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), event_type, json.dumps(details) if details else None)
        )

def _redact_secrets(data: str) -> str:
    """Redact sensitive info from logs (IMP-008)."""
    if not data:
        return data
    data = re.sub(r'sk-[A-Za-z0-9]{20,}', 'sk-...[REDACTED]', data)
    data = re.sub(r'nvapi-[A-Za-z0-9_\-]{20,}', 'nvapi-...[REDACTED]', data)
    data = re.sub(r'AKIA[0-9A-Z]{16}', 'AKIA...[REDACTED]', data)
    data = re.sub(r'(base64,[a-zA-Z0-9+/=]{1000})[a-zA-Z0-9+/=]+', r'\1...[TRUNCATED]', data)
    return data

def insert_log(entry: dict):
    """Insert a log row into the DB (thread-safe)."""
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
    if "request_body" in entry and isinstance(entry["request_body"], str):
        entry["request_body"] = _redact_secrets(entry["request_body"])
    if "response_body" in entry and isinstance(entry["response_body"], str):
        entry["response_body"] = _redact_secrets(entry["response_body"])
        
    with get_db() as conn:
        conn.execute(sql, entry)

def update_log_tokens(request_id: str, prompt: int, completion: int, response_body: str = None):
    """Update token counts after a streaming response finishes, with optional full response."""
    with get_db() as conn:
        if response_body:
            response_body = _redact_secrets(response_body)
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
# Token extraction helpers
# ═══════════════════════════════════════════════════════════════

def extract_tokens_from_response(body: dict) -> Tuple[int, int]:
    """Pull token counts from a non-streaming JSON response."""
    usage = body.get("usage") or {}
    return usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)

def extract_tokens_from_chunks(chunks: List[Dict]) -> Tuple[int, int]:
    """
    Extract token counts from collected SSE chunks.
    Token usage is typically in the last chunk's `usage` field.
    """
    for chunk in reversed(chunks):
        usage = chunk.get("usage")
        if usage:
            return usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    return 0, 0

# ═══════════════════════════════════════════════════════════════
# OpenAI compatibility helpers
# ═══════════════════════════════════════════════════════════════

def _authorized() -> bool:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return False
    return hmac.compare_digest(auth[7:], config["GATEWAY_API_KEY"])

def _openai_error(message: str, error_type: str = "invalid_request_error", status_code: int = 400) -> Tuple[Dict, int]:
    """Create standardized OpenAI error response."""
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": None,
            "code": None
        }
    }, status_code

def _validate_messages(messages: List[Dict]) -> Optional[str]:
    """Basic message validation for OpenAI format."""
    if not isinstance(messages, list) or not messages:
        return "messages must be a non-empty array"
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            return f"message {i} must be an object"
        role = msg.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            return f"message {i} has invalid role '{role}' (must be system, user, assistant, or tool)"
        if "content" not in msg and "tool_calls" not in msg:
            return f"message {i} must have content or tool_calls"
        if role == "tool" and not msg.get("tool_call_id"):
            return f"tool message {i} requires tool_call_id"
        if role == "assistant" and msg.get("tool_calls"):
            content = msg.get("content")
            if content not in (None, ""):
                return f"assistant message {i} with tool_calls must have empty content"
    return None

def _enforce_model_allowlist(req_body: dict) -> Tuple[str, str]:
    """Enforce ALLOWED_MODELS list and return (requested, used). (IMP-021)"""
    allowed_models = config.get("ALLOWED_MODELS", "*")
    requested = req_body.get("model", config["CUSTOM_MODEL_ID"])
    used = requested
    if allowed_models != "*":
        allowed_list = [m.strip() for m in allowed_models.split(",")]
        if requested not in allowed_list:
            used = config["CUSTOM_MODEL_ID"]
            
    req_body["model"] = used
    return requested, used

def _retry_post(url, **kwargs):
    """Idempotent POST with retries (IMP-016)."""
    retries = config["UPSTREAM_RETRIES"]
    for attempt in range(retries + 1):
        try:
            resp = upstream_session.post(url, **kwargs)
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep((2 ** attempt) * 0.5 + 0.1)
                continue
            return resp
        except requests.RequestException as e:
            if attempt < retries:
                time.sleep((2 ** attempt) * 0.5 + 0.1)
                continue
            raise e

def _retry_get(url, **kwargs):
    """Idempotent GET with retries (IMP-016)."""
    retries = config["UPSTREAM_RETRIES"]
    for attempt in range(retries + 1):
        try:
            resp = upstream_session.get(url, **kwargs)
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep((2 ** attempt) * 0.5 + 0.1)
                continue
            return resp
        except requests.RequestException as e:
            if attempt < retries:
                time.sleep((2 ** attempt) * 0.5 + 0.1)
                continue
            raise e

def accumulate_tool_calls(chunks: List[Dict]) -> List[Dict]:
    """
    Accumulate parallel tool calls from streaming chunks.
    Handles multiple concurrent tool calls by index.
    """
    tool_calls_by_index: Dict[int, Dict] = {}

    for chunk in chunks:
        choices = chunk.get("choices", [])
        for choice in choices:
            delta = choice.get("delta", {})
            if "tool_calls" not in delta:
                continue

            for tool_call in delta["tool_calls"]:
                index = tool_call.get("index", 0)

                if index not in tool_calls_by_index:
                    tool_calls_by_index[index] = {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""}
                    }

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
    """Extract full text content from accumulated chunks."""
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
    """Add CORS headers to all responses for OpenAI SDK compatibility."""
    origins = config.get("GATEWAY_CORS_ORIGINS", "*")
    origin = request.headers.get("Origin", "")
    if origins == "*":
        response.headers.add('Access-Control-Allow-Origin', '*')
    else:
        allowed = [o.strip() for o in origins.split(",")]
        if origin in allowed:
            response.headers.add('Access-Control-Allow-Origin', origin)
            response.headers.add('Vary', 'Origin')
        else:
            response.headers.add('Access-Control-Allow-Origin', allowed[0])
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization, Accept, X-Session-ID')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    response.headers.add('Access-Control-Max-Age', '86400')
    return response

def _log_error(entry: dict, start_ts: float, status: int, msg: str):
    """Persist an error log row."""
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
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit (NIT-11)
_cors_origins = config.get("GATEWAY_CORS_ORIGINS", "*")
CORS(app, resources={r"/*": {"origins": _cors_origins.split(",") if _cors_origins != "*" else "*"}})

@app.after_request
def after_request(response: Response) -> Response:
    """Add CORS headers to all responses."""
    return add_cors_headers(response)

# ═══════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════

@app.route("/", methods=["GET", "OPTIONS"])
@app.route("/v1", methods=["GET", "OPTIONS"])
def root_endpoint():
    """Root endpoint for health checks and discovery."""
    if request.method == "OPTIONS":
        resp = Response("", status=200)
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
            "gateway": {
                "status": "/gateway/status",
                "stats": "/gateway/stats",
                "logs": "/gateway/logs"
            }
        }
    })

@app.route("/v1/chat/completions", methods=["POST", "OPTIONS"])
@app.route("/chat/completions",    methods=["POST", "OPTIONS"])
def chat_completions():
    if request.method == "OPTIONS":
        resp = Response("", status=200)
        return add_cors_headers(resp)

    if not _authorized():
        logger.warning("Unauthorized from %s", request.remote_addr)
        error, code = _openai_error("Invalid API key", "invalid_api_key", 401)
        resp = jsonify(error)
        resp.status_code = code
        return add_cors_headers(resp)

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

    original_model, used_model = _enforce_model_allowlist(req_body)

    is_stream = bool(req_body.get("stream", False))
    request_id = str(uuid.uuid4())
    start_ts   = time.time()

    log_entry = {
        "request_id":      request_id,
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "endpoint":        "/v1/chat/completions",
        "method":          "POST",
        "client_ip":       request.remote_addr,
        "model_requested": original_model,
        "model_used":      used_model,
        "streaming":       int(is_stream),
        "request_body":    json.dumps(req_body)[:16000] if config["LOG_REQUEST_BODIES"] else None,
        "response_status": 0,
        "response_body":   None,
        "prompt_tokens":   0,
        "completion_tokens": 0,
        "total_tokens":    0,
        "duration_ms":     0,
        "error":           None,
    }

    target_url = f"{config['CUSTOM_BASE_URL']}/chat/completions"
    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {config['CUSTOM_API_KEY']}",
        "User-Agent":    "NVIDIA-AI-Gateway/2.0",
        "Accept":        "text/event-stream" if is_stream else "application/json",
    }

    if is_stream:
        # BUG-019: Ensure upstream returns usage in final chunk
        req_body.setdefault("stream_options", {}).setdefault("include_usage", True)
        MAX_STREAMING_CHUNKS = int(os.getenv("MAX_STREAMING_CHUNKS", "5000"))
        collected_chunks = []

        try:
            upstream = upstream_session.post(
                target_url,
                json=req_body,
                headers=headers,
                stream=True,
                timeout=(config["UPSTREAM_CONNECT_TIMEOUT"], config["UPSTREAM_READ_TIMEOUT"]),
            )
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
                upstream_json = json.loads(body)
                err_resp = upstream_json
            except Exception:
                err_resp, _ = _openai_error(f"Upstream error: {body[:200]}", "api_error", upstream.status_code)
            resp = jsonify(err_resp)
            resp.status_code = upstream.status_code
            return add_cors_headers(resp)

        log_entry["response_status"] = 200
        insert_log(log_entry)

        def generate():
            nonlocal log_entry, start_ts, collected_chunks, request_id
            try:
                for raw_line in upstream.iter_lines(chunk_size=None, decode_unicode=True):
                    if not raw_line:
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
                upstream.close()  # BUG-007: Always close upstream connection
                duration_ms = (time.time() - start_ts) * 1000
                prompt_tok, compl_tok = extract_tokens_from_chunks(collected_chunks)
                full_response = extract_full_content(collected_chunks)
                tool_calls = accumulate_tool_calls(collected_chunks)
                response_for_log = {"content": full_response, "tool_calls": tool_calls}
                try:
                    update_log_tokens(request_id, prompt_tok, compl_tok,
                                      response_body=json.dumps(response_for_log)[:16000])
                    with get_db() as conn:
                        conn.execute(
                            "UPDATE api_logs SET duration_ms=? WHERE request_id=?",
                            (round(duration_ms, 2), request_id),
                        )
                except Exception as db_err:
                    logger.error("DB update after stream failed: %s", db_err)

                logger.info(
                    "[%s] stream done | tokens p=%d c=%d | %.0f ms",
                    request_id[:8], prompt_tok, compl_tok, duration_ms,
                )

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
        resp = upstream_session.post(
            target_url,
            json=req_body, headers=headers,
            timeout=(config["UPSTREAM_CONNECT_TIMEOUT"], config["UPSTREAM_READ_TIMEOUT"]),
        )
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

        logger.info(
            "[%s] non-stream | status=%d tokens p=%d c=%d | %.0f ms",
            request_id[:8], resp.status_code, prompt_tok, compl_tok, duration_ms,
        )

        response = Response(
            resp.content,
            status=resp.status_code,
            content_type=resp.headers.get("Content-Type", "application/json"),
            headers={"X-Gateway-RequestId": request_id}
        )
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
@app.route("/completions",    methods=["POST", "OPTIONS"])
def completions():
    if request.method == "OPTIONS":
        resp = Response("", status=200)
        return add_cors_headers(resp)

    if not _authorized():
        logger.warning("Unauthorized from %s", request.remote_addr)
        error, code = _openai_error("Invalid API key", "invalid_api_key", 401)
        resp = jsonify(error)
        resp.status_code = code
        return add_cors_headers(resp)

    try:
        req_body = request.get_json(force=True)
    except Exception:
        error, code = _openai_error("Invalid JSON body", "invalid_request_error", 400)
        return add_cors_headers(jsonify(error)), code

    if not req_body:
        error, code = _openai_error("Empty request body", "invalid_request_error", 400)
        return add_cors_headers(jsonify(error)), code

    # BUG-011: Validate prompt field
    if "prompt" not in req_body:
        error, code = _openai_error("Missing required field 'prompt'", "invalid_request_error", 400)
        return add_cors_headers(jsonify(error)), code

    original_model, used_model = _enforce_model_allowlist(req_body)

    is_stream = bool(req_body.get("stream", False))
    request_id = str(uuid.uuid4())
    start_ts = time.time()

    log_entry = {
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": "/v1/completions",
        "method": "POST",
        "client_ip": request.remote_addr,
        "model_requested": original_model,
        "model_used": used_model,
        "streaming": int(is_stream),
        "request_body": json.dumps(req_body)[:16000] if config["LOG_REQUEST_BODIES"] else None,
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
        # BUG-019: Ensure upstream returns usage in final chunk
        req_body.setdefault("stream_options", {}).setdefault("include_usage", True)
        MAX_STREAMING_CHUNKS = int(os.getenv("MAX_STREAMING_CHUNKS", "5000"))
        collected_chunks = []

        try:
            upstream = upstream_session.post(target_url, json=req_body, headers=headers, stream=True, timeout=(config["UPSTREAM_CONNECT_TIMEOUT"], config["UPSTREAM_READ_TIMEOUT"]))
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
                upstream_json = json.loads(body)
                err_resp = upstream_json
            except Exception:
                err_resp, _ = _openai_error(f"Upstream error: {body[:200]}", "api_error", upstream.status_code)
            resp = jsonify(err_resp)
            resp.status_code = upstream.status_code
            return add_cors_headers(resp)

        log_entry["response_status"] = 200
        insert_log(log_entry)

        def generate():
            nonlocal log_entry, start_ts, collected_chunks, request_id
            try:
                for raw_line in upstream.iter_lines(chunk_size=None, decode_unicode=True):
                    if not raw_line:
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
                upstream.close()  # BUG-007: Always close upstream connection
                duration_ms = (time.time() - start_ts) * 1000
                prompt_tok, compl_tok = extract_tokens_from_chunks(collected_chunks)
                full_response = extract_full_content(collected_chunks)
                tool_calls = accumulate_tool_calls(collected_chunks)
                response_for_log = {"content": full_response, "tool_calls": tool_calls}
                try:
                    update_log_tokens(request_id, prompt_tok, compl_tok,
                                      response_body=json.dumps(response_for_log)[:16000])
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
            resp = upstream_session.post(target_url, json=req_body, headers=headers, timeout=(config["UPSTREAM_CONNECT_TIMEOUT"], config["UPSTREAM_READ_TIMEOUT"]))
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
@app.route("/embeddings",    methods=["POST", "OPTIONS"])
def embeddings():
    if request.method == "OPTIONS":
        resp = Response("", status=200)
        return add_cors_headers(resp)

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

    original_model, used_model = _enforce_model_allowlist(req_body)

    request_id = str(uuid.uuid4())
    start_ts = time.time()

    log_entry = {
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": "/v1/embeddings",
        "method": "POST",
        "client_ip": request.remote_addr,
        "model_requested": original_model,
        "model_used":      used_model,
        "streaming":       0,     "request_body": json.dumps(req_body)[:16000] if config["LOG_REQUEST_BODIES"] else None,
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
        resp = _retry_post(target_url, json=req_body, headers=headers, timeout=(config["UPSTREAM_CONNECT_TIMEOUT"], config["UPSTREAM_READ_TIMEOUT"]))
        duration_ms = (time.time() - start_ts) * 1000

        try:
            resp_body = resp.json()
        except Exception:
            resp_body = {"raw": resp.text}

        # BUG-009: Use upstream usage when available; embeddings have no completion tokens
        prompt_tokens = 0
        completion_tokens = 0
        if isinstance(resp_body, dict) and "usage" in resp_body:
            prompt_tokens = resp_body["usage"].get("prompt_tokens", 0)
            completion_tokens = 0  # Embeddings never produce completion tokens

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
@app.route("/models",    methods=["GET", "OPTIONS"])
def list_models():
    if request.method == "OPTIONS":
        resp = Response("", status=200)
        return add_cors_headers(resp)

    if not _authorized():
        error, code = _openai_error("Invalid API key", "invalid_api_key", 401)
        resp = jsonify(error)
        resp.status_code = code
        return add_cors_headers(resp)

    try:
        resp = _retry_get(
            f"{config['CUSTOM_BASE_URL']}/models",
            headers={"Authorization": f"Bearer {config['CUSTOM_API_KEY']}"},
            timeout=15,
        )
        if resp.status_code == 200:
            response = Response(resp.content, status=200,
                                content_type=resp.headers.get("Content-Type", "application/json"),
                                headers={"X-Models-Source": "upstream"})
            return add_cors_headers(response)
        else:
            logger.warning("[models] Upstream returned status=%d: %s", resp.status_code, resp.text[:200])
    except Exception as e:
        logger.warning("[models] Upstream request failed: %s", e)

    fallback = jsonify({
        "object": "list",
        "data": [{
            "id": config["CUSTOM_MODEL_ID"],
            "object": "model",
            "created": int(time.time()),
            "owned_by": "nvidia",
        }],
    })
    fallback.headers["X-Models-Source"] = "fallback"
    return add_cors_headers(fallback)

@app.route("/healthz", methods=["GET"])
def healthz():
    return Response("OK", status=200)

@app.route("/readyz", methods=["GET"])
def readyz():
    # Check DB
    try:
        with get_db() as conn:
            conn.execute("SELECT 1 FROM api_logs LIMIT 1")
    except Exception as e:
        logger.error("readyz: DB error: %s", e)
        return Response("DB Error", status=503)

    # Check upstream
    try:
        resp = upstream_session.get(
            f"{config['CUSTOM_BASE_URL']}/models",
            headers={"Authorization": f"Bearer {config['CUSTOM_API_KEY']}"},
            timeout=2.0,
        )
        if resp.status_code != 200:
            logger.error("readyz: Upstream returned %d", resp.status_code)
            return Response("Upstream Error", status=503)
    except Exception as e:
        logger.error("readyz: Upstream error: %s", e)
        return Response("Upstream Error", status=503)

    return Response("Ready", status=200)

@app.route("/gateway/status", methods=["GET", "OPTIONS"])
def gateway_status():
    if request.method == "OPTIONS":
        resp = Response("", status=200)
        return add_cors_headers(resp)

    return add_cors_headers(jsonify({
        "status": "running",
        "gateway_port": config["GATEWAY_PORT"],
        "target_base_url": config["CUSTOM_BASE_URL"],
        "target_model": config["CUSTOM_MODEL_ID"],
        "db_path": config["DB_PATH"],
        "features": [
            "OpenAI-compatible streaming",
            "SQLite logging",
            "CORS enabled",
            "Tool calls support",
            "Embeddings endpoint",
            "Completions endpoint"
        ]
    }))

@app.route("/gateway/stats", methods=["GET", "OPTIONS"])
def gateway_stats():
    if request.method == "OPTIONS":
        resp = Response("", status=200)
        return add_cors_headers(resp)

    if not _authorized():
        error, code = _openai_error("Invalid API key", "invalid_api_key", 401)
        resp = jsonify(error)
        resp.status_code = code
        return add_cors_headers(resp)

    with get_db() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*)                          AS total_requests,
                SUM(streaming)                    AS streaming_requests,
                SUM(total_tokens)                 AS total_tokens,
                SUM(prompt_tokens)                AS total_prompt_tokens,
                SUM(completion_tokens)            AS total_completion_tokens,
                AVG(duration_ms)                  AS avg_duration_ms,
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
        resp = Response("", status=200)
        return add_cors_headers(resp)

    if not _authorized():
        error, code = _openai_error("Invalid API key", "invalid_api_key", 401)
        resp = jsonify(error)
        resp.status_code = code
        return add_cors_headers(resp)

    limit  = request.args.get("limit",  50,  type=int)
    offset = request.args.get("offset", 0,   type=int)
    stream_only = request.args.get("stream", None)

    where = ""
    params: List = []
    if stream_only is not None:
        where = "WHERE streaming = ?"
        params.append(1 if stream_only.lower() in ("1", "true") else 0)

    with get_db() as conn:
        total = conn.execute(f"SELECT COUNT(*) FROM api_logs {where}", params).fetchone()[0]
        rows  = conn.execute(
            f"SELECT * FROM api_logs {where} ORDER BY id DESC LIMIT ? OFFSET ?",
            params + [limit, offset],
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

    return add_cors_headers(jsonify({
        "total": total,
        "limit": limit,
        "offset": offset,
        "returned": len(logs),
        "logs": logs,
    }))

# ═══════════════════════════════════════════════════════════════
# Background server
# ═══════════════════════════════════════════════════════════════
class BackgroundServer:
    def __init__(self, flask_app, host, port):
        self.server = make_server(host, port, flask_app)
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        """Gracefully shut down the server (BUG-002)."""
        self.server.shutdown()
        self.server.server_close()
        self._thread.join(timeout=5)


# Shutdown event for responsive heartbeat sleep (BUG-013)
_shutdown_event = threading.Event()

def log_retention_worker():
    """Background worker to delete old logs and vacuum DB (IMP-019)."""
    while not _shutdown_event.is_set():
        try:
            with get_db() as conn:
                days = config["LOG_RETENTION_DAYS"]
                cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
                cursor = conn.execute("DELETE FROM api_logs WHERE timestamp < ?", (cutoff,))
                deleted = cursor.rowcount
                if deleted > 0:
                    logger.info("Log retention worker deleted %d old rows", deleted)
                    
            # Run once a day
            _shutdown_event.wait(86400)
        except Exception as e:
            logger.error("Error in log retention worker: %s", e)
            _shutdown_event.wait(3600)

        

def _local_ip() -> str:
    """Get local IP address. Try hostname first, then UDP probe with timeout (NIT-9)."""
    try:
        ip = socket.gethostbyname(socket.gethostname())
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(2)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _get_version() -> str:
    """Get version from package metadata (NIT-1)."""
    try:
        from importlib.metadata import version
        return version("nvidia-ai-gateway")
    except Exception:
        return "2.0.0"

# ═══════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════
def main():
    """Main entry point for the gateway."""
    parser = argparse.ArgumentParser(description="NVIDIA AI Gateway - OpenAI-Compatible Proxy")
    parser.add_argument("--host", type=str, default=None, help="Host to bind to (default: 127.0.0.1, env: GATEWAY_HOST)")
    parser.add_argument("--port", type=int, default=None, help="Port to run the gateway on (default: 8089, env: GATEWAY_PORT)")
    parser.add_argument("--config", type=str, help="Path to config file (INI format)")
    parser.add_argument("--log-level", type=str, default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    args = parser.parse_args()

    # BUG-018: Load config file BEFORE validation so --config can supply CUSTOM_API_KEY
    if args.config:
        import configparser
        cfg = configparser.ConfigParser()
        cfg.read(args.config)
        if "gateway" in cfg:
            for key in ["CUSTOM_BASE_URL", "CUSTOM_MODEL_ID", "CUSTOM_API_KEY", "DB_PATH", "LOG_FILE",
                         "GATEWAY_HOST", "GATEWAY_CORS_ORIGINS"]:
                if key in cfg["gateway"]:
                    config[key] = cfg["gateway"][key]
        if "server" in cfg and "port" in cfg["server"]:
            config["GATEWAY_PORT"] = int(cfg["server"]["port"])

    # BUG-022: Override config only when CLI args are explicitly provided (default=None)
    if args.port is not None:
        config["GATEWAY_PORT"] = args.port
    if args.host is not None:
        config["GATEWAY_HOST"] = args.host
    if args.log_level is not None:
        config["LOG_LEVEL"] = args.log_level
        logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    # Validate (BUG-018: now runs after config file merge)
    if not config["CUSTOM_API_KEY"]:
        print("ERROR: CUSTOM_API_KEY is not set!", file=sys.stderr)
        print("Set via environment variable, config file (--config), or CUSTOM_API_KEY env var.", file=sys.stderr)
        sys.exit(1)

    # Initialize DB
    init_db()
    insert_audit_log("startup", {"version": _get_version(), "port": config["GATEWAY_PORT"]})
    logger.info("SQLite DB ready at %s", config["DB_PATH"])

    ip = _local_ip()
    gateway_version = _get_version()
    host = config["GATEWAY_HOST"]
    port = config["GATEWAY_PORT"]
    base = f"http://{ip}:{port}"

    # BUG-014: Mask API key in banner/logs (show only first 8 chars)
    masked_key = config["GATEWAY_API_KEY"][:8] + "..."

    BANNER = f"""
╔══════════════════════════════════════════════════════════════════╗
║       NVIDIA AI Gateway v{gateway_version}  (OpenAI-Compatible)          ║
╠══════════════════════════════════════════════════════════════════╣
║  Bind Address     : {host}:{port}
║  Gateway Base URL : {base}/v1
║  Gateway API Key  : {masked_key}
║  Target URL       : {config['CUSTOM_BASE_URL']}
║  Target Model     : {config['CUSTOM_MODEL_ID']}
║  DB               : {config['DB_PATH']}
╠══════════════════════════════════════════════════════════════════╣
║  OPENAI SDK EXAMPLE
╠══════════════════════════════════════════════════════════════════╣

    from openai import OpenAI

    client = OpenAI(
        base_url="{base}/v1",
        api_key="<your-gateway-api-key>"
    )

    # Streaming chat completion
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
║  AVAILABLE ENDPOINTS
╠══════════════════════════════════════════════════════════════════╣
║  POST {base}/v1/chat/completions  (streaming + non-streaming)
║  POST {base}/v1/completions       (legacy completions)
║  POST {base}/v1/embeddings
║  GET  {base}/v1/models
║  GET  {base}/gateway/status
║  GET  {base}/gateway/stats
║  GET  {base}/gateway/logs?limit=50&offset=0
║  OPTIONS any endpoint (CORS preflight)
╚══════════════════════════════════════════════════════════════════╝
    """
    print(BANNER)

    # BUG-001/003: Pass host to BackgroundServer
    server = BackgroundServer(app, host, port)
    server.start()

    # Start retention worker (IMP-019)
    retention_thread = threading.Thread(target=log_retention_worker, daemon=True)
    retention_thread.start()

    # BUG-001: Signal handlers as closures capturing `server`
    def handle_shutdown(signum, frame):
        logger.info("Shutdown signal received...")
        _shutdown_event.set()
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)
    logger.info("Gateway running on %s:%d — press Ctrl+C to stop", host, port)

    # BUG-013: Use Event.wait() for responsive shutdown, MAX(id) for O(1) heartbeat
    try:
        while not _shutdown_event.is_set():
            _shutdown_event.wait(60)
            if _shutdown_event.is_set():
                break
            with get_db() as conn:
                row = conn.execute("SELECT MAX(id) FROM api_logs").fetchone()
                total = row[0] if row[0] is not None else 0
            logger.info("Heartbeat — total logged requests: %d", total)
    except KeyboardInterrupt:
        logger.info("Shutting down …")
        server.stop()
        logger.info("Gateway stopped")

if __name__ == "__main__":
    main()
