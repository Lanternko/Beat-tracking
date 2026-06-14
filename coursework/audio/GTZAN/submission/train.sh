#!/bin/bash
# Train all models for GTZAN genre classification
# Usage: bash train.sh

set -e

echo "=== Training CNN (best model, ~3 min) ==="
python train.py \
  --csv_path gtzan.csv \
  --audio_root ./genres \
  --out_dir ./checkpoint_best \
  --cache_dir ./feature_cache \
  --epochs 300 \
  --patience 50 \
  --samples_per_song 10 \
  --num_workers 4

echo ""
echo "=== Training CNN with SpecAugment (~3 min) ==="
python train.py \
  --csv_path gtzan.csv \
  --audio_root ./genres \
  --out_dir ./checkpoint_augment \
  --cache_dir ./feature_cache \
  --epochs 300 \
  --patience 50 \
  --samples_per_song 10 \
  --num_workers 4 \
  --augment \
  --freq_mask_param 10 \
  --time_mask_param 10

echo ""
echo "=== Training XGBoost baseline ==="
python train_xgb.py \
  --csv_path gtzan.csv \
  --audio_root ./genres \
  --out_dir ./checkpoint_xgb \
  --cache_dir ./feature_cache_xgb \
  --max_depth 4 \
  --n_estimators 3000 \
  --learning_rate 0.03 \
  --subsample 0.7 \
  --colsample 0.7

echo ""
echo "=== All training complete ==="
