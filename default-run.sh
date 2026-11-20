#!/usr/bin/env bash
# default-run.sh - score revprefs.csv and write revprefs.scored.<timestamp>.csv for upload to HotCRP.
#
#   ./default-run.sh                          specter2+scincl (OpenReview-style) on the GPU
#   ./default-run.sh rerank                   another method
#   ./default-run.sh specter2 cpu             another device: gpu (default), cpu, cuda:1, mps
#   ./default-run.sh rerank gpu other.csv     another input file
#   ./default-run.sh rerank gpu revprefs.csv --keep-original   extra score_bids.py flags after the file
#
# "gpu" lets score_bids.py pick whatever GPU there is (CUDA, then Apple MPS) and falls back to
# the CPU, with a note, when there is none.
set -eu
cd "$(dirname "$0")"
METHOD=${1:-specter2+scincl}
DEVICE=${2:-gpu}
INPUT=${3:-revprefs.csv}
[ "$DEVICE" = gpu ] && DEVICE=auto
if [ -z "${PYTHON:-}" ]; then                    # PYTHON=... to pick the interpreter
    PYTHON=python3
    [ -x .venv/bin/python ] && PYTHON=.venv/bin/python
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
exec "$PYTHON" score_bids.py "$INPUT" --method "$METHOD" --device "$DEVICE" "${@:4}"
