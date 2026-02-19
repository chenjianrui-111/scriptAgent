#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <user@host> <ssh_key_path> [remote_dir] [ssh_port]"
  echo "Example: $0 root@47.99.119.146 ~/Downloads/scriptAgent.pem /opt/script-agent 22"
  exit 1
fi

REMOTE="$1"
SSH_KEY="$2"
REMOTE_DIR="${3:-/opt/script-agent}"
SSH_PORT="${4:-22}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"

if [[ ! -f "${SSH_KEY}" ]]; then
  echo "ERROR: ssh key not found: ${SSH_KEY}"
  exit 2
fi

if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git command not found"
  exit 2
fi

if ! command -v scp >/dev/null 2>&1; then
  echo "ERROR: scp command not found"
  exit 2
fi

if ! command -v ssh >/dev/null 2>&1; then
  echo "ERROR: ssh command not found"
  exit 2
fi

cd "${REPO_ROOT}"

COMMIT_SHA="$(git rev-parse --short HEAD)"
RELEASE_TS="$(date +%Y%m%d-%H%M%S)"
ARCHIVE_NAME="script-agent-${RELEASE_TS}-${COMMIT_SHA}.tar.gz"
ARCHIVE_PATH="/tmp/${ARCHIVE_NAME}"

cleanup() {
  rm -f "${ARCHIVE_PATH}"
}
trap cleanup EXIT

echo "[1/5] Build release archive from git HEAD: ${COMMIT_SHA}"
git archive --format=tar.gz -o "${ARCHIVE_PATH}" HEAD

echo "[2/5] Upload archive to remote server"
scp -P "${SSH_PORT}" -i "${SSH_KEY}" "${ARCHIVE_PATH}" "${REMOTE}:/tmp/${ARCHIVE_NAME}"

echo "[3/5] Extract release to ${REMOTE_DIR}"
ssh -p "${SSH_PORT}" -i "${SSH_KEY}" "${REMOTE}" "\
set -euo pipefail; \
mkdir -p '${REMOTE_DIR}'; \
tar -xzf '/tmp/${ARCHIVE_NAME}' -C '${REMOTE_DIR}'; \
if [[ ! -f '${REMOTE_DIR}/.env' ]]; then \
  if [[ -f '${REMOTE_DIR}/.env.production.example' ]]; then \
    cp '${REMOTE_DIR}/.env.production.example' '${REMOTE_DIR}/.env'; \
  elif [[ -f '${REMOTE_DIR}/.env.example' ]]; then \
    cp '${REMOTE_DIR}/.env.example' '${REMOTE_DIR}/.env'; \
  fi; \
fi; \
rm -f '/tmp/${ARCHIVE_NAME}'"

echo "[4/5] Rebuild and restart service"
ssh -p "${SSH_PORT}" -i "${SSH_KEY}" "${REMOTE}" "\
cd '${REMOTE_DIR}' && \
COMPOSE_BAKE=false DOCKER_BUILDKIT=0 docker compose -f docker-compose.prod.yml up -d --build"

echo "[5/5] Health check"
ssh -p "${SSH_PORT}" -i "${SSH_KEY}" "${REMOTE}" "\
set -euo pipefail; \
for i in 1 2 3 4 5 6; do \
  if curl -sS --max-time 15 http://127.0.0.1:8080/api/v1/health; then \
    exit 0; \
  fi; \
  sleep 2; \
done; \
echo 'ERROR: health check failed after retries'; \
exit 1"

echo
echo "Release success:"
echo "  commit: ${COMMIT_SHA}"
echo "  remote: ${REMOTE}:${REMOTE_DIR}"
echo
echo "Quick checks:"
echo "  ssh -p ${SSH_PORT} -i ${SSH_KEY} ${REMOTE} 'cd ${REMOTE_DIR} && docker compose -f docker-compose.prod.yml ps'"
echo "  curl http://<server-ip>:8080/api/v1/health"
