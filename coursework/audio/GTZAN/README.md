# Music Genre Classification — GTZAN Dataset

CNN-based music genre classifier using log-mel spectrograms with hybrid pooling and temporal ensemble inference.

## Environment

```bash
pip install -r requirements.txt
```

**Dependencies**: torch, numpy, pandas, librosa, soundfile, scipy, numba, audioread

**Hardware used**: NVIDIA RTX 5090 (33 GB VRAM)

## Dataset

The GTZAN dataset CSV (`gtzan.csv`) has the format:

```
ID,label,set
00776.au,disco,train
00507.au,blues,val
00123.au,jazz,test
```

- Train: 800 songs | Val: 100 songs | Test: 100 songs
- 10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock

Audio files should be placed in `./genres/` directory.

## Training

### Basic training (best single model)

```bash
python train_cnn.py \
  --csv_path gtzan.csv \
  --audio_root ./genres \
  --out_dir ./checkpoint_cnn_v4 \
  --cache_dir ./feature_cache_cnn_v4 \
  --epochs 300 \
  --patience 50 \
  --samples_per_song 10
```

### Training with SpecAugment

```bash
python train_cnn.py \
  --csv_path gtzan.csv \
  --audio_root ./genres \
  --out_dir ./checkpoint_cnn_v5_augment \
  --cache_dir ./feature_cache_cnn_v4 \
  --epochs 300 \
  --patience 50 \
  --samples_per_song 10 \
  --augment \
  --freq_mask_param 10 \
  --time_mask_param 10
```

### Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 200 | Maximum training epochs |
| `--patience` | 30 | Early stopping patience |
| `--samples_per_song` | 10 | Random crops per song per epoch |
| `--augment` | off | Enable SpecAugment |
| `--freq_mask_param` | 10 | SpecAugment frequency mask width |
| `--time_mask_param` | 10 | SpecAugment time mask width |
| `--t0` | 30 | CosineAnnealingWarmRestarts T_0 |
| `--top_db` | 20 | Silence trimming threshold (dB) |

## Inference

### Single model inference

```bash
# Validation (to check accuracy)
python predict_cnn.py \
  --csv_path gtzan.csv \
  --audio_root ./genres \
  --ckpt_path ./checkpoint_cnn_v4/checkpoint_best.pt \
  --out_csv val_pred.csv \
  --predict_set val \
  --n_segments 10

# Test set (for Kaggle submission)
python predict_cnn.py \
  --csv_path gtzan.csv \
  --audio_root ./genres \
  --ckpt_path ./checkpoint_cnn_v4/checkpoint_best.pt \
  --out_csv submission.csv \
  --n_segments 10
```

### Ensemble inference (soft voting)

```bash
# Validation
python predict_ensemble.py \
  --csv_path gtzan.csv \
  --audio_root ./genres \
  --ckpt_paths ./checkpoint_cnn_v4_long/checkpoint_best.pt \
               ./checkpoint_cnn_v5_augment/checkpoint_best.pt \
  --out_csv val_ensemble.csv \
  --predict_set val \
  --n_segments 10

# Test set (for Kaggle submission)
python predict_ensemble.py \
  --csv_path gtzan.csv \
  --audio_root ./genres \
  --ckpt_paths ./checkpoint_cnn_v4_long/checkpoint_best.pt \
               ./checkpoint_cnn_v5_augment/checkpoint_best.pt \
  --out_csv submission_ensemble.csv \
  --n_segments 10
```

## Results

| Model | Val Accuracy | Parameters | Best Epoch |
|-------|-------------|------------|------------|
| CNN v1 (fixed first-3s) | 91% | 102K | 47 |
| CNN v2 (random crop, small) | 72% | 102K | 45 |
| CNN v4 (deep + hybrid pooling) | 94% | 1.1M | 67 |
| CNN v4_long (patience=50) | 94% | 1.1M | 58 |
| CNN v5_augment (SpecAugment) | 94% | 1.1M | 48 |
| Ensemble (v4_long + v5_augment) | 94% | 2.2M | — |

## File Structure

```
├── train_cnn.py           # Training script (CNN v4)
├── predict_cnn.py         # Single model inference
├── predict_ensemble.py    # Ensemble inference (soft voting)
├── gtzan.csv              # Dataset split file
├── genres/                # Audio files (.au)
├── requirements.txt       # Dependencies
└── README.md              # This file
```
