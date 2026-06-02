# NeMo Guardrails server. NVIDIA does not publish a pre-built public image,
# so we build a slim one from the upstream pip package.
FROM python:3.11-slim

# nemoguardrails pulls in annoy (a C-extension nearest-neighbor lib) which
# needs gcc + python headers. Install just enough to compile, then purge.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential python3-dev \
 && pip install --no-cache-dir "nemoguardrails[server]>=0.10.0" "openai>=1.40.0" \
 && apt-get purge -y --auto-remove build-essential python3-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
EXPOSE 8090

# The config dir is bind-mounted by docker-compose at /app/config.
# nemoguardrails 0.10 CLI doesn't accept --host (binds 0.0.0.0 by default).
CMD ["nemoguardrails", "server", "--config", "/app/config", "--port", "8090"]
