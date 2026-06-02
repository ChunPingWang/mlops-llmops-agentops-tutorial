# NeMo Guardrails server. NVIDIA does not publish a pre-built public image,
# so we build a slim one from the upstream pip package.
FROM python:3.11-slim

RUN pip install --no-cache-dir "nemoguardrails>=0.10.0" "openai>=1.40.0"

WORKDIR /app
EXPOSE 8090

# The config dir is bind-mounted by docker-compose at /app/config.
CMD ["nemoguardrails", "server", "--config", "/app/config", "--port", "8090", "--host", "0.0.0.0"]
