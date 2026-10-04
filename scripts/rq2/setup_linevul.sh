#!/usr/bin/env bash
# Install the exact public LineVul implementation used by the RQ2 adapter.
set -euo pipefail

DEST="${1:-third_party/LineVul}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
COMMIT="9401ec6e60b762a9308645961268203244bee048"

if [[ ! -d "${DEST}/.git" ]]; then
  git clone https://github.com/awsm-research/LineVul.git "${DEST}"
fi
git -C "${DEST}" fetch --depth 1 origin "${COMMIT}"
git -C "${DEST}" checkout --detach "${COMMIT}"

"${PYTHON_BIN}" -m venv "${DEST}/.venv"
"${DEST}/.venv/bin/python" -m pip install --upgrade pip
"${DEST}/.venv/bin/python" -m pip install -r "${DEST}/requirements.txt"

echo "LineVul ${COMMIT} installed at ${DEST}"
