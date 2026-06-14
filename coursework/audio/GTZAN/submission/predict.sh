#!/bin/bash
# Generate predictions for GTZAN genre classification
# Usage: bash predict.sh

set -e

echo "=== Ensemble inference (best submission) ==="
python predict_ensemble.py \
  --csv_path gtzan.csv \
  --audio_root ./genres \
  --ckpt_paths ./checkpoint_best/checkpoint_best.pt \
              ./checkpoint_augment/checkpoint_best.pt \
  --out_csv submission_ensemble.csv \
  --n_segments 10

echo ""
echo "=== Single model inference ==="
python predict.py \
  --csv_path gtzan.csv \
  --audio_root ./genres \
  --ckpt_path ./checkpoint_best/checkpoint_best.pt \
  --out_csv submission_single.csv \
  --n_segments 10

echo ""
echo "=== All predictions complete ==="
echo "Submit submission_ensemble.csv to Kaggle"
