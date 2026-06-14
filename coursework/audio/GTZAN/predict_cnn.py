#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
predict_cnn.py  —  CNN v4 + Temporal Ensemble 推論

模型架構與 train_cnn.py v4 完全一致：
  channels: 64→128→256→256, Hybrid GAP+GMP Pooling

Temporal Ensemble：
  將 30 秒音訊切成 n_segments 個均勻片段，
  各自取 softmax 機率後平均，再 argmax 決定最終類別。

Usage:
  # 驗證集（確認效果）
  python predict_cnn.py \\
    --csv_path gtzan.csv \\
    --audio_root ./genres \\
    --ckpt_path ./checkpoint_cnn_v4/checkpoint_best.pt \\
    --out_csv val_pred_v4.csv \\
    --predict_set val \\
    --n_segments 10

  # 測試集（上傳 Kaggle）
  python predict_cnn.py \\
    --csv_path gtzan.csv \\
    --audio_root ./genres \\
    --ckpt_path ./checkpoint_cnn_v4/checkpoint_best.pt \\
    --out_csv submission_cnn_v4.csv \\
    --n_segments 10
"""

import os
import argparse
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1) Model（必須與 train_cnn.py v4 完全一致）
# ============================================================
class MelCNN(nn.Module):
    """
    升級版 CNN v4：
      Block 1: Conv(64)  + BN + ReLU + MaxPool(2x2) + Dropout(0.2)
      Block 2: Conv(128) + BN + ReLU + MaxPool(2x2) + Dropout(0.2)
      Block 3: Conv(256) + BN + ReLU + MaxPool(2x4) + Dropout(0.3)
      Block 4: Conv(256) + BN + ReLU + MaxPool(2x4) + Dropout(0.3)
      Hybrid Pooling: GAP + GMP → concat(512) → Dense(256) → Dense(num_classes)
    """
    def __init__(self, num_classes=10):
        super().__init__()

        def conv_block(in_ch, out_ch, pool_size, dropout_p):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(pool_size),
                nn.Dropout2d(dropout_p),
            )

        self.block1 = conv_block(1,   64,  (2, 2), 0.2)
        self.block2 = conv_block(64,  128, (2, 2), 0.2)
        self.block3 = conv_block(128, 256, (2, 4), 0.3)
        self.block4 = conv_block(256, 256, (2, 4), 0.3)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        avg = self.gap(x)
        mx  = self.gmp(x)
        x   = torch.cat([avg, mx], dim=1)
        return self.classifier(x)


# ============================================================
# 2) 單段特徵提取
# ============================================================
def extract_mel_segment(y, sr, start_frame, n_mels, n_frames, hop_length, n_fft):
    start_sample = start_frame * hop_length
    end_sample   = start_sample + (n_frames + 1) * hop_length + n_fft

    segment = y[start_sample:end_sample]
    if len(segment) < n_fft:
        segment = np.pad(segment, (0, n_fft - len(segment)))

    mel = librosa.feature.melspectrogram(
        y=segment, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft
    )
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

    T = mel_db.shape[1]
    if T < n_frames:
        mel_db = np.pad(mel_db, ((0, 0), (0, n_frames - T)))
    else:
        mel_db = mel_db[:, :n_frames]

    mu  = mel_db.mean()
    std = mel_db.std() + 1e-6
    mel_db = (mel_db - mu) / std

    return mel_db[np.newaxis, :, :]  # (1, n_mels, n_frames)


# ============================================================
# 3) Temporal Ensemble 推論
# ============================================================
def predict_with_temporal_ensemble(
    file_path, model, device, idx2label,
    n_segments=10, n_mels=128, n_frames=128,
    hop_length=512, n_fft=2048, target_sr=22050
):
    y, sr = librosa.load(file_path, sr=target_sr)

    total_frames = max(1, len(y) // hop_length - n_frames)
    start_frames = np.linspace(0, total_frames, n_segments, dtype=int)

    probs_list = []
    model.eval()
    with torch.no_grad():
        for start_f in start_frames:
            seg = extract_mel_segment(
                y, sr, start_f,
                n_mels=n_mels, n_frames=n_frames,
                hop_length=hop_length, n_fft=n_fft
            )
            x      = torch.from_numpy(seg).unsqueeze(0).to(device)
            logits = model(x)
            probs  = F.softmax(logits, dim=1).cpu().numpy()
            probs_list.append(probs[0])

    avg_probs = np.mean(probs_list, axis=0)
    pred_idx  = int(np.argmax(avg_probs))
    return idx2label[pred_idx]


# ============================================================
# 4) Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path",    type=str, required=True)
    ap.add_argument("--audio_root",  type=str, required=True)
    ap.add_argument("--ckpt_path",   type=str, required=True)
    ap.add_argument("--out_csv",     type=str, default="submission_cnn.csv")
    ap.add_argument("--n_segments",  type=int, default=10)
    ap.add_argument("--n_mels",      type=int, default=128)
    ap.add_argument("--n_frames",    type=int, default=128)
    ap.add_argument("--hop_length",  type=int, default=512)
    ap.add_argument("--n_fft",       type=int, default=2048)
    ap.add_argument("--target_sr",   type=int, default=22050)
    ap.add_argument("--predict_set", type=str, default="test")
    ap.add_argument("--fail_on_missing", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")
    print(f"[Temporal Ensemble] n_segments = {args.n_segments}")

    # 載入 checkpoint
    ckpt = torch.load(args.ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "idx2label" in ckpt:
        idx2label   = {int(k): v for k, v in ckpt["idx2label"].items()}
        num_classes = len(idx2label)
    else:
        raise ValueError("Checkpoint 缺少 idx2label。")

    # 建立模型
    model = MelCNN(num_classes=num_classes).to(device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[Model] loaded from {args.ckpt_path}  classes={num_classes}")

    # 讀 CSV
    df = pd.read_csv(args.csv_path)
    df.columns = [c.strip() for c in df.columns]
    df["ID"]  = df["ID"].astype(str).str.strip()
    df["set"] = df["set"].astype(str).str.strip().str.lower()

    pred_df = df[df["set"] == args.predict_set].copy()
    if len(pred_df) == 0:
        raise ValueError(f"找不到 set=={args.predict_set} 的資料。")
    print(f"[Data] 共 {len(pred_df)} 筆 ({args.predict_set} set)")

    # 逐一推論
    preds     = []
    correct   = 0
    has_label = ("label" in df.columns) and (args.predict_set == "val")

    for i, (_, row) in enumerate(pred_df.iterrows(), 1):
        audio_id = row["ID"]
        path     = os.path.join(args.audio_root, audio_id)

        if not os.path.isfile(path):
            if args.fail_on_missing:
                raise FileNotFoundError(f"找不到音訊檔：{path}")
            pred_label = idx2label[0]
        else:
            pred_label = predict_with_temporal_ensemble(
                path, model, device, idx2label,
                n_segments = args.n_segments,
                n_mels     = args.n_mels,
                n_frames   = args.n_frames,
                hop_length = args.hop_length,
                n_fft      = args.n_fft,
                target_sr  = args.target_sr,
            )

        preds.append((audio_id, pred_label))

        if has_label:
            true_label = str(row["label"]).strip()
            if pred_label == true_label:
                correct += 1
            else:
                print(f"  [X] {audio_id}: true={true_label}, pred={pred_label}")

        if i % 10 == 0:
            print(f"  ... {i}/{len(pred_df)} done")

    if has_label:
        acc = correct / len(preds) * 100
        print(f"\n[Val Accuracy] {correct}/{len(preds)} = {acc:.2f}%")

    out = pd.DataFrame(preds, columns=["ID", "label"])
    out.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"[OK] saved: {args.out_csv}  rows={len(out)}")


if __name__ == "__main__":
    main()
