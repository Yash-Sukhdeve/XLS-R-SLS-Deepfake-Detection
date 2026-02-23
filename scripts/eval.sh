#!/usr/bin/env bash
set -euo pipefail
# Evaluate a trained model on DF, LA, and In-the-Wild tracks.
#
# Usage:
#   bash scripts/eval.sh <model.pth> [database_base_path]
#
# Example:
#   bash scripts/eval.sh models/model_DF_WCE_50_5_1e-06/epoch_2.pth /media/data

MODEL=${1:?"Usage: bash scripts/eval.sh <model.pth> [database_base_path]"}
DB_BASE=${2:-""}

echo "=== Evaluating model: $MODEL ==="

# --- DF track ---
echo ""
echo "--- ASVspoof 2021 DF ---"
DF_DB=${DB_BASE:+"$DB_BASE/ASVspoof2021_DF_eval/"}
DF_DB=${DF_DB:-""}
if [ -n "$DF_DB" ]; then
    rm -f scores/scores_DF_eval.txt
    python -m sls_asvspoof.train --track=DF --is_eval --eval \
        --model_path="$MODEL" \
        --protocols_path=database/ASVspoof_DF_cm_protocols/ASVspoof2021.DF.cm.eval.trl.txt \
        --database_path="$DF_DB" \
        --eval_output=scores/scores_DF_eval.txt
    python -m sls_asvspoof.evaluate --track DF scores/scores_DF_eval.txt ./keys eval
fi

# --- LA track ---
echo ""
echo "--- ASVspoof 2021 LA ---"
LA_DB=${DB_BASE:+"$DB_BASE/ASVspoof2021_LA_eval/"}
LA_DB=${LA_DB:-""}
if [ -n "$LA_DB" ]; then
    rm -f scores/scores_LA_eval.txt
    python -m sls_asvspoof.train --track=LA --is_eval --eval \
        --model_path="$MODEL" \
        --protocols_path=database/ASVspoof_DF_cm_protocols/ASVspoof2021.LA.cm.eval.trl.txt \
        --database_path="$LA_DB" \
        --eval_output=scores/scores_LA_eval.txt
    python -m sls_asvspoof.evaluate --track LA scores/scores_LA_eval.txt ./keys eval
fi

# --- In-the-Wild track ---
echo ""
echo "--- In-the-Wild ---"
WILD_DB=${DB_BASE:+"$DB_BASE/release_in_the_wild/"}
WILD_DB=${WILD_DB:-""}
if [ -n "$WILD_DB" ]; then
    rm -f scores/scores_Wild_eval.txt
    python -m sls_asvspoof.train --track=In-the-Wild --is_eval --eval \
        --model_path="$MODEL" \
        --protocols_path=database/ASVspoof_DF_cm_protocols/in_the_wild.eval.txt \
        --database_path="$WILD_DB" \
        --eval_output=scores/scores_Wild_eval.txt
    python -m sls_asvspoof.evaluate --track Wild scores/scores_Wild_eval.txt ./keys eval
fi

echo ""
echo "=== Evaluation complete ==="
