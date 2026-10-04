#!/usr/bin/env bash
# Synthesize VMAVul samples for one training dataset and draw the RQ2 augmentation set.
#   usage: scripts/run_synthesis.sh <megavul|primevul|icvul|titanvul>
# Requires normalized dataset JSONL, the validation toolchain, and the model endpoints
# documented in README.md.
set -euo pipefail
TRAIN_SET="$1"
export VMAVUL_TRAIN_SET="${TRAIN_SET}"
python scripts/data/build_reference_pool.py --train-set "${TRAIN_SET}"
python -m vmavul synthesize --config configs/vmavul.yaml --output-dir "runs/${TRAIN_SET}/vmavul" \
  --reference-pool "data/reference_pool/${TRAIN_SET}.jsonl"
python -m vmavul export --run "runs/${TRAIN_SET}/vmavul" --out "runs/${TRAIN_SET}/vmavul/samples.jsonl"
python scripts/sample_augmentation.py --input "runs/${TRAIN_SET}/vmavul/samples.jsonl" \
  --out "data/augmentation/${TRAIN_SET}/vmavul.jsonl" --n 15000 --seed 0
