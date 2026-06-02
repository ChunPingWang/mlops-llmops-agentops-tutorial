# Python runner image — used by scripts/run_poc.sh to execute the
# integration tests and agent/rag/mlops scripts inside the compose network.
FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential python3-dev curl git \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app
CMD ["python", "--version"]
