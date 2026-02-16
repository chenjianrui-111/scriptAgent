#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <user@host> [remote_dir] [ssh_port]"
  exit 1
fi

REMOTE="$1"
REMOTE_DIR="${2:-/opt/script-agent}"
SSH_PORT="${3:-22}"

echo "[1/4] Preparing remote directory: ${REMOTE_DIR}"
ssh -p "${SSH_PORT}" "${REMOTE}" "mkdir -p '${REMOTE_DIR}'"

echo "[2/4] Syncing project files"
rsync -az --delete \
  --exclude ".git" \
  --exclude "__pycache__" \
  --exclude ".pytest_cache" \
  --exclude "tests" \
  --exclude ".claude" \
  -e "ssh -p ${SSH_PORT}" \
  ./ "${REMOTE}:${REMOTE_DIR}/"

echo "[3/4] Preparing .env on remote"
ssh -p "${SSH_PORT}" "${REMOTE}" \
  "cd '${REMOTE_DIR}' && if [[ ! -f .env ]]; then cp .env.production.example .env; fi"

echo "[3.5/4] Validating remote .env"
ssh -p "${SSH_PORT}" "${REMOTE}" "cd '${REMOTE_DIR}' && bash -lc '
if grep -q \"^LLM_FALLBACK_BACKEND=zhipu\" .env \
  && grep -q \"^ZHIPU_API_KEY=replace_with_real_key\" .env; then
  echo \"ERROR: ZHIPU_API_KEY is still placeholder in .env\"
  exit 2
fi
'"

echo "[4/4] Building and starting service"
ssh -p "${SSH_PORT}" "${REMOTE}" \
  "cd '${REMOTE_DIR}' && docker compose -f docker-compose.prod.yml up -d --build"

echo "Deploy finished."
echo "Check status:"
echo "  ssh -p ${SSH_PORT} ${REMOTE} \"cd ${REMOTE_DIR} && docker compose -f docker-compose.prod.yml ps\""
echo "Health probe:"
echo "  curl http://<server-ip>:8080/api/v1/health"
