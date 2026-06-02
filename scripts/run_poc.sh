#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# run_poc.sh — End-to-end PoC launcher and verifier.
#
# Brings up the 14-service Docker Compose stack, waits for healthchecks,
# seeds Qdrant, then runs the integration tests and every script in
# scripts/ + agents/ + rag/ + mlops/ in order, recording PASS/FAIL/SKIP.
#
# Usage:
#   bash scripts/run_poc.sh [--no-bring-up] [--no-tear-down] [--only <step>]
#
# Prereqs (see README):
#   * docker + docker compose installed and daemon running
#   * Python 3.11+ with pip
#   * .env created from .env.example with real secrets
#   * Local Gemma 4 endpoint already serving on :8000 (or expect V-01 to skip)
# -----------------------------------------------------------------------------
set -uo pipefail

# Resolve repo root regardless of where the script is invoked from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Colors
RED='\033[31m'; GREEN='\033[32m'; YELLOW='\033[33m'; BLUE='\033[34m'; RESET='\033[0m'

# ─── Argument parsing ────────────────────────────────────────────────────────
BRING_UP=1
TEAR_DOWN=0    # default: leave stack up so user can inspect
ONLY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-bring-up)   BRING_UP=0; shift ;;
    --no-tear-down)  TEAR_DOWN=0; shift ;;
    --tear-down)     TEAR_DOWN=1; shift ;;
    --only)          ONLY="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

# ─── Result tracker (printed at the end) ─────────────────────────────────────
declare -a RESULTS=()
record() {  # record <step-name> <status> [detail]
  local name="$1" status="$2" detail="${3:-}"
  RESULTS+=("${name}|${status}|${detail}")
  case "$status" in
    PASS) echo -e "  ${GREEN}[PASS]${RESET} ${name} ${detail}" ;;
    FAIL) echo -e "  ${RED}[FAIL]${RESET} ${name} ${detail}" ;;
    SKIP) echo -e "  ${YELLOW}[SKIP]${RESET} ${name} ${detail}" ;;
  esac
}

run_step() {  # run_step <name> <log-file> -- <cmd...>
  local name="$1" log="$2"; shift 2
  [[ "$1" == "--" ]] && shift
  if [[ -n "${ONLY}" && "${ONLY}" != "${name}" ]]; then
    record "${name}" "SKIP" "(filtered by --only)"
    return 0
  fi
  echo -e "${BLUE}▶ ${name}${RESET}  (log: ${log})"
  if "$@" >"${log}" 2>&1; then
    record "${name}" "PASS"
  else
    record "${name}" "FAIL" "exit=$? — see ${log}"
  fi
}

# ─── 0. Preflight ────────────────────────────────────────────────────────────
echo -e "${BLUE}── Preflight ──${RESET}"
if ! command -v docker >/dev/null 2>&1; then
  echo -e "${RED}docker not found in PATH.${RESET}"; exit 1
fi
if ! docker compose version >/dev/null 2>&1; then
  echo -e "${RED}docker compose v2 plugin not found.${RESET}"; exit 1
fi
if [[ ! -f .env ]]; then
  if [[ -f .env.example ]]; then
    echo -e "${YELLOW}No .env found — copying .env.example. Edit it before re-running for real keys.${RESET}"
    cp .env.example .env
  else
    echo -e "${RED}Neither .env nor .env.example exists.${RESET}"; exit 1
  fi
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo -e "${RED}python3 not found.${RESET}"; exit 1
fi

# Python deps for the verification scripts (best-effort; user-managed venv preferred)
if [[ ! -d .venv ]]; then
  echo -e "${BLUE}Creating .venv ...${RESET}"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
echo -e "${BLUE}Installing/upgrading Python deps (this may take a few minutes) ...${RESET}"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt python-dotenv httpx \
  || { echo -e "${RED}pip install failed${RESET}"; exit 1; }

# ─── 1. Bring up the stack ───────────────────────────────────────────────────
if [[ "${BRING_UP}" == "1" ]]; then
  echo -e "${BLUE}── Bringing up Docker Compose stack ──${RESET}"
  if ! docker compose up -d 2>&1 | tee "${LOG_DIR}/compose-up.log"; then
    record "stack/compose-up" "FAIL" "see ${LOG_DIR}/compose-up.log"
  else
    record "stack/compose-up" "PASS"
  fi

  # ─── 2. Wait for healthchecks (max 5 min) ───────────────────────────────────
  echo -e "${BLUE}── Waiting for services to become healthy (max 5 min) ──${RESET}"
  deadline=$(( $(date +%s) + 300 ))
  while :; do
    # Collect health states; any container without a healthcheck is treated as
    # 'running' once it's up.
    ps_json="$(docker compose ps --format json 2>/dev/null || true)"
    unhealthy="$(echo "${ps_json}" | python3 -c '
import json, sys
data = sys.stdin.read().strip()
if not data:
    sys.exit(0)
# docker compose ps emits NDJSON in newer versions, a JSON array in older
try:
    rows = json.loads(data)
    if isinstance(rows, dict):
        rows = [rows]
except json.JSONDecodeError:
    rows = [json.loads(l) for l in data.splitlines() if l.strip()]
bad = []
for r in rows:
    name = r.get("Name") or r.get("Service") or "?"
    state = (r.get("State") or "").lower()
    health = (r.get("Health") or "").lower()
    if state not in ("running",):
        bad.append(f"{name}={state}")
    elif health and health not in ("healthy", ""):
        bad.append(f"{name}=health:{health}")
print(",".join(bad))
')"
    if [[ -z "${unhealthy}" ]]; then
      record "stack/healthy" "PASS"
      break
    fi
    if (( $(date +%s) >= deadline )); then
      record "stack/healthy" "FAIL" "still unhealthy: ${unhealthy}"
      break
    fi
    sleep 5
  done

  docker compose ps > "${LOG_DIR}/compose-ps.log" 2>&1 || true
fi

# ─── 3. Seed Qdrant ──────────────────────────────────────────────────────────
run_step "scripts/seed_qdrant.py"   "${LOG_DIR}/seed_qdrant.log"   -- python scripts/seed_qdrant.py

# ─── 4. Integration test suite (V-01..V-25) ──────────────────────────────────
run_step "scripts/test_integration" "${LOG_DIR}/test_integration.log" -- python scripts/test_integration.py

# ─── 5. Agents ───────────────────────────────────────────────────────────────
run_step "agents/react_agent.py"    "${LOG_DIR}/react_agent.log"    -- python agents/react_agent.py
run_step "agents/rag_agent.py"      "${LOG_DIR}/rag_agent.log"      -- python agents/rag_agent.py
run_step "agents/hitl_agent.py"     "${LOG_DIR}/hitl_agent.log"     -- python agents/hitl_agent.py
run_step "agents/mem0_agent.py"     "${LOG_DIR}/mem0_agent.log"     -- python agents/mem0_agent.py

# ─── 6. RAG pipeline + evaluation ────────────────────────────────────────────
# rag_pipeline.py needs at least one .txt or .pdf under rag/data/; seed one
mkdir -p rag/data
if [[ ! -s rag/data/sample.txt ]]; then
  cat > rag/data/sample.txt <<'EOF'
RAG (Retrieval-Augmented Generation) augments LLM responses with external
documents retrieved from a vector database. Common stacks pair LangChain or
LlamaIndex with Qdrant, Pinecone, or Weaviate. Embeddings are computed by
sentence-transformer or OpenAI-style embedding endpoints and indexed for
approximate nearest-neighbor search.
EOF
fi
run_step "rag/rag_pipeline.py"      "${LOG_DIR}/rag_pipeline.log"   -- python rag/rag_pipeline.py
run_step "rag/eval_ragas.py"        "${LOG_DIR}/eval_ragas.log"     -- python rag/eval_ragas.py

# ─── 7. MLOps ────────────────────────────────────────────────────────────────
run_step "mlops/mlflow_experiment.py" "${LOG_DIR}/mlflow_experiment.log" -- python mlops/mlflow_experiment.py
run_step "mlops/dagster_pipeline.py"  "${LOG_DIR}/dagster_pipeline.log"  -- python mlops/dagster_pipeline.py

# ─── 8. Summary ──────────────────────────────────────────────────────────────
echo
echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BLUE}║                  PoC verification summary                         ║${RESET}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════════╝${RESET}"
pass=0; fail=0; skip=0
for line in "${RESULTS[@]}"; do
  IFS='|' read -r name status detail <<<"${line}"
  case "${status}" in
    PASS) pass=$((pass+1)); printf "  ${GREEN}%-6s${RESET} %s\n" "${status}" "${name}" ;;
    FAIL) fail=$((fail+1)); printf "  ${RED}%-6s${RESET} %s  %s\n" "${status}" "${name}" "${detail}" ;;
    SKIP) skip=$((skip+1)); printf "  ${YELLOW}%-6s${RESET} %s  %s\n" "${status}" "${name}" "${detail}" ;;
  esac
done
echo
echo -e "  Total: ${GREEN}${pass} PASS${RESET}  ${RED}${fail} FAIL${RESET}  ${YELLOW}${skip} SKIP${RESET}"
echo -e "  Logs:  ${LOG_DIR}/"

# ─── 9. Tear down (opt-in) ───────────────────────────────────────────────────
if [[ "${TEAR_DOWN}" == "1" ]]; then
  echo -e "${BLUE}Tearing down stack ...${RESET}"
  docker compose down -v
fi

exit "$(( fail == 0 ? 0 : 1 ))"
