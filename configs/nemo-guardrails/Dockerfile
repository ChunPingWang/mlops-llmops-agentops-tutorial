# NeMo Guardrails server. NVIDIA does not publish a pre-built public image,
# so we build a slim one from the upstream pip package.
FROM python:3.11-slim

# nemoguardrails pulls in annoy (a C-extension nearest-neighbor lib) which
# needs gcc + python headers. gettext-base provides envsubst (used by
# entrypoint to expand ${VAR} in config.yml, since NeMo Guardrails itself
# doesn't substitute env vars in YAML).
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential python3-dev gettext-base \
 && pip install --no-cache-dir "nemoguardrails[server]>=0.10.0" "openai>=1.40.0" \
 && apt-get purge -y --auto-remove build-essential python3-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
EXPOSE 8090

# Entry: bind-mounted /app/config has YAML with ${VAR} refs. envsubst
# expands them into /app/config-rendered (read-only mount stays clean),
# then exec nemoguardrails pointing at the rendered dir.
RUN mkdir -p /app/config-rendered/rails
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
