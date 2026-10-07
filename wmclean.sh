#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
if [[ ! -x "$ROOT/.venv/bin/wmclean" ]]; then
  echo "wmclean is not installed yet. Run ./setup_linux.sh gpu (or core/cpu-ai) first." >&2
  exit 2
fi
exec "$ROOT/.venv/bin/wmclean" "$@"
