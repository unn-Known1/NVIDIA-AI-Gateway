FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim
WORKDIR /app

# Create non-root user (NIT-4)
RUN groupadd -r gateway && useradd -r -g gateway gateway

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY . .

# Exclude DB/log files from image (they're runtime artifacts)
RUN rm -f gateway_requests.db gateway.log

RUN chown -R gateway:gateway /app
USER gateway

EXPOSE 8089

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8089/gateway/status')" || exit 1

CMD ["python", "-m", "gateway"]
