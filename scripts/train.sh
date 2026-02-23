#!/usr/bin/env bash
set -euo pipefail
# Train SLS+XLS-R model on ASVspoof2019 LA (DF track config)
# Override any config value via CLI: bash scripts/train.sh --batch_size 10
python -m sls_asvspoof.train --config configs/train_df.yaml "$@"
