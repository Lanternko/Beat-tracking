# Music Genre Classification — GTZAN Dataset

CNN-based music genre classifier using log-mel spectrograms with hybrid pooling and temporal ensemble inference.

## Environment

```bash
pip install -r requirements.txt
```

**Dependencies**: torch, numpy, pandas, librosa, soundfile, scipy, numba, audioread, xgboost

**Hardware used**: NVIDIA RTX 5090 (33 GB VRAM)

## Dataset

Audio files (`.au`) should be placed in `./genres/`. The split file `gtzan.csv` has the format:

```
ID,label,set
00776.au,disco,train
00507.au,blues,val
00123.au,jazz,test
```

800 train / 100 val / 100 test, 10 genres.

## Quick Start

```bash
# Train + predict in one go:
bash train.sh
bash predict.sh
```

## Training

### CNN (best model)

```bash
python train.py \
  --csv_path gtzan.csv \
  --audio_root ./genres \
  --out_dir ./checkpoint_best \
  --cache_dir ./feature_cache \
  --epochs 300 \
  --patience 50 \
  --samples_per_song 10 \
  --num_workers 4
```

### CNN with SpecAugment

```bash
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
```

### XGBoost baseline

```bash
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
```

## Inference

### Single CNN model

```bash
python predict.py \
  --csv_path gtzan.csv \
  --audio_root ./genres \
  --ckpt_path ./checkpoint_best/checkpoint_best.pt \
  --out_csv submission.csv \
  --n_segments 10
```

### Ensemble (soft voting, 2 models)

```bash
python predict_ensemble.py \
  --csv_path gtzan.csv \
  --audio_root ./genres \
  --ckpt_paths ./checkpoint_best/checkpoint_best.pt \
              ./checkpoint_augment/checkpoint_best.pt \
  --out_csv submission_ensemble.csv \
  --n_segments 10
```

### XGBoost

```bash
python predict_xgb.py \
  --csv_path gtzan.csv \
  --audio_root ./genres \
  --ckpt_path ./checkpoint_xgb/checkpoint_best.pt \
  --out_csv submission_xgb.csv
```

## Results

| Model | Val Accuracy | Kaggle Public |
|-------|-------------|---------------|
| LSTM baseline (provided) | ~70% | — |
| XGBoost (handcrafted features) | 86% | — |
| CNN v1 (fixed first-3s, 102K params) | 91% | 1.000 |
| CNN v2 (random crop, 102K params) | 72% | — |
| CNN v3 (deep + hybrid pooling, 1.1M params) | 94% | 1.000 |
| Ensemble (v3_long + v3_augment) | 94% | 1.000 |

## File Structure

```
├── train.py              # CNN training script
├── predict.py            # CNN inference (temporal ensemble)
├── predict_ensemble.py   # Multi-model soft voting inference
├── train_xgb.py          # XGBoost training script
├── predict_xgb.py        # XGBoost inference
├── train.sh              # One-command training
├── predict.sh            # One-command inference
├── requirements.txt      # Dependencies
└── README.md             # This file
```
