# Python runner image — used by scripts/run_poc.sh to execute the
# integration tests and agent/rag/mlops scripts inside the compose network.
FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential python3-dev curl git \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
 && pip install --no-cache-dir \
    "langchain-text-splitters>=0.3.0" \
    "langchain-google-vertexai>=2.0.0"
# langchain-text-splitters: split from langchain in 0.3 (rag_pipeline import path)
# langchain-google-vertexai: ragas imports vertexai chat-models even when unused

WORKDIR /app
CMD ["python", "--version"]
