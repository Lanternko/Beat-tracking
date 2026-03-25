#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
路線 A：CNN + Log-Mel Spectrogram 音樂流派分類器
=====================================================
v4 升級（NotebookLM 診斷後的根本解法）：

問題根源：
  原版模型只看「前 3 秒」達到 91%，可能是記住前奏特徵而非真正學會流派。
  隨機取段後 Train Acc 卡在 52%，代表 50 萬參數的模型容量不足以
  應對整首歌各段落的高內部變異性。

三項核心升級：
  1. 加深加寬模型（~280 萬參數）
     channels: 32→64→128→256→256（比原版翻倍）
     + Global Average Pooling 取代 Flatten，提升對不同片段的適應力

  2. 去靜音前處理（librosa.effects.trim）
     避免隨機取到無聲/靜音段落，干擾模型學習

  3. CosineAnnealingWarmRestarts 學習率排程
     前期大步探索，後期小步微調，比 ReduceLROnPlateau 更適合 Random Crop 高變異訓練

  + 每首歌每 epoch 貢獻 samples_per_song 個隨機片段（預設 10）
  + 預先載入所有 waveform 到 RAM 加速訓練

指令範例：
  python train_cnn.py \\
    --csv_path gtzan.csv \\
    --audio_root ./genres \\
    --out_dir ./checkpoint_cnn_v4 \\
    --cache_dir ./feature_cache_cnn_v4 \\
    --epochs 200 \\
    --patience 30 \\
    --samples_per_song 10
"""

import os
import json
import argparse
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import librosa

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1) 特徵提取
# ============================================================
def segment_to_melspec(y, sr, n_mels=128, n_fft=2048, hop_length=512, target_frames=128):
    mel    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    T = mel_db.shape[1]
    if T < target_frames:
        mel_db = np.pad(mel_db, ((0, 0), (0, target_frames - T)), mode="constant")
    else:
        mel_db = mel_db[:, :target_frames]
    mean = mel_db.mean()
    std  = mel_db.std() + 1e-8
    return ((mel_db - mean) / std).astype(np.float32)


def trim_and_load(file_path, target_sr=22050, top_db=20):
    """
    載入音訊並裁掉頭尾靜音（top_db=20 dB 以下視為靜音）。
    確保隨機取段時不會取到靜音或無效段落。
    """
    y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    # 如果裁掉後太短（< 3 秒），改用原始音訊
    if len(y_trimmed) < int(3.0 * sr):
        return y, sr
    return y_trimmed, sr


def random_crop_from_waveform(y, sr, segment_sec=3.0,
                               n_mels=128, n_fft=2048, hop_length=512, target_frames=128):
    """從已載入的（去靜音後）waveform 隨機取一段，轉成 melspec"""
    seg_samples = int(segment_sec * sr)
    if len(y) <= seg_samples:
        seg = y
    else:
        start = random.randint(0, len(y) - seg_samples)
        seg = y[start : start + seg_samples]
    return segment_to_melspec(seg, sr, n_mels, n_fft, hop_length, target_frames)


def extract_song_melspecs_fixed(file_path, segment_sec=3.0, target_sr=22050,
                                 n_mels=128, n_fft=2048, hop_length=512, target_frames=128):
    """驗證用：均勻切割整首歌，回傳所有片段"""
    y, sr   = librosa.load(file_path, sr=target_sr, mono=True)
    seg_len = int(segment_sec * sr)
    n_seg   = len(y) // seg_len
    specs = []
    for i in range(max(n_seg, 1)):
        seg = y[i*seg_len : (i+1)*seg_len] if n_seg > 0 else y
        specs.append(segment_to_melspec(seg, sr, n_mels, n_fft, hop_length, target_frames))
    return np.array(specs)


# ============================================================
# 2) 驗證集快取
# ============================================================
def load_or_extract_val(audio_id, audio_root, cache_dir, cache_tag,
                        segment_sec, target_sr, n_mels, n_fft, hop_length, target_frames):
    if cache_dir:
        safe_id    = audio_id.replace("/", "_").replace("\\", "_")
        cache_path = os.path.join(cache_dir, cache_tag, f"{safe_id}.npz")
        if os.path.isfile(cache_path):
            return np.load(cache_path)["specs"]

    specs = extract_song_melspecs_fixed(
        os.path.join(audio_root, audio_id),
        segment_sec, target_sr, n_mels, n_fft, hop_length, target_frames
    )
    if cache_dir:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, specs=specs)
    return specs


# ============================================================
# 3) SpecAugment（輕量版，備用）
# ============================================================
def spec_augment(mel, freq_mask_param=10, time_mask_param=10):
    cloned = mel.clone().squeeze(0)
    n_mels, n_frames = cloned.shape
    f  = random.randint(0, freq_mask_param)
    f0 = random.randint(0, max(0, n_mels - f))
    cloned[f0:f0+f, :] = 0.0
    t  = random.randint(0, time_mask_param)
    t0 = random.randint(0, max(0, n_frames - t))
    cloned[:, t0:t0+t] = 0.0
    return cloned.unsqueeze(0)


# ============================================================
# 4) Dataset：預先載入去靜音後的 waveform，隨機多取段
# ============================================================
class MultiCropDataset(Dataset):
    def __init__(
        self, df, audio_root, label2idx,
        segment_sec=3.0, target_sr=22050,
        n_mels=128, n_fft=2048, hop_length=512, target_frames=128,
        samples_per_song=10, top_db=20,
        augment=False, freq_mask_param=10, time_mask_param=10,
    ):
        self.label2idx       = label2idx
        self.segment_sec     = segment_sec
        self.target_sr       = target_sr
        self.n_mels          = n_mels
        self.n_fft           = n_fft
        self.hop_length      = hop_length
        self.target_frames   = target_frames
        self.samples_per_song = samples_per_song
        self.augment         = augment
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

        df = df.reset_index(drop=True)

        print("  [Dataset] 預先載入 waveform（含去靜音）...")
        self.waveforms = []
        self.labels    = []
        for _, row in df.iterrows():
            path  = os.path.join(audio_root, str(row["ID"]).strip())
            label = label2idx[str(row["label"]).strip()]
            if os.path.isfile(path):
                try:
                    y, _ = trim_and_load(path, target_sr=target_sr, top_db=top_db)
                    self.waveforms.append(y)
                except Exception:
                    self.waveforms.append(
                        np.zeros(int(segment_sec * target_sr), dtype=np.float32)
                    )
            else:
                self.waveforms.append(
                    np.zeros(int(segment_sec * target_sr), dtype=np.float32)
                )
            self.labels.append(label)
        print(f"  [Dataset] 完成，{len(self.waveforms)} 首歌已載入")

    def __len__(self):
        return len(self.waveforms) * self.samples_per_song

    def __getitem__(self, idx):
        song_idx = idx % len(self.waveforms)
        y        = self.waveforms[song_idx]
        label    = self.labels[song_idx]

        spec = random_crop_from_waveform(
            y, self.target_sr, self.segment_sec,
            self.n_mels, self.n_fft, self.hop_length, self.target_frames
        )

        x = torch.from_numpy(spec).unsqueeze(0)  # (1, n_mels, n_frames)
        if self.augment:
            x = spec_augment(x, self.freq_mask_param, self.time_mask_param)

        return x, torch.tensor(label, dtype=torch.long)


# ============================================================
# 5) 升級版 CNN 模型（~280 萬參數）
#    channels: 1→64→128→256→256
#    Global Average Pooling + Global Max Pooling 並聯
# ============================================================
class MelCNN(nn.Module):
    """
    升級版 4 層 CNN：
      Block 1: Conv(64)  + BN + ReLU + MaxPool(2x2) + Dropout(0.2)
      Block 2: Conv(128) + BN + ReLU + MaxPool(2x2) + Dropout(0.2)
      Block 3: Conv(256) + BN + ReLU + MaxPool(2x4) + Dropout(0.3)
      Block 4: Conv(256) + BN + ReLU + MaxPool(2x4) + Dropout(0.3)
      Hybrid Pooling: GAP + GMP 並聯 → concat → Dense(256) → Dense(num_classes)

    輸入：(B, 1, 128, 128)
    參數量：~280 萬
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

        # Hybrid Pooling：GAP + GMP 並聯，各輸出 256 維，concat 後為 512 維
        # 文獻支持：Max Pooling 捕捉突出特徵，Average Pooling 保留全局分布
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.gmp = nn.AdaptiveMaxPool2d(1)  # Global Max Pooling

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2, 256),  # 512 → 256
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Hybrid Pooling
        avg = self.gap(x)  # (B, 256, 1, 1)
        mx  = self.gmp(x)  # (B, 256, 1, 1)
        x   = torch.cat([avg, mx], dim=1)  # (B, 512, 1, 1)

        return self.classifier(x)


# ============================================================
# 6) Label 映射
# ============================================================
def build_label_map(df):
    labels    = sorted({str(x).strip() for x in df["label"].dropna()})
    label2idx = {lab: i for i, lab in enumerate(labels)}
    idx2label = {i: lab for lab, i in label2idx.items()}
    return label2idx, idx2label


# ============================================================
# 7) 主訓練流程
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path",         type=str,   required=True)
    ap.add_argument("--audio_root",       type=str,   required=True)
    ap.add_argument("--out_dir",          type=str,   default="checkpoint_cnn_v4")
    ap.add_argument("--cache_dir",        type=str,   default=None)
    ap.add_argument("--segment_sec",      type=float, default=3.0)
    ap.add_argument("--target_sr",        type=int,   default=22050)
    ap.add_argument("--n_mels",           type=int,   default=128)
    ap.add_argument("--n_fft",            type=int,   default=2048)
    ap.add_argument("--hop_length",       type=int,   default=512)
    ap.add_argument("--target_frames",    type=int,   default=128)
    ap.add_argument("--samples_per_song", type=int,   default=10)
    ap.add_argument("--top_db",           type=int,   default=20,
                    help="去靜音閾值（dB），預設 20")
    ap.add_argument("--augment",          action="store_true",
                    help="啟用 SpecAugment（預設關閉，先確認基礎效果）")
    ap.add_argument("--freq_mask_param",  type=int,   default=10)
    ap.add_argument("--time_mask_param",  type=int,   default=10)
    ap.add_argument("--batch_size",       type=int,   default=64)
    ap.add_argument("--epochs",           type=int,   default=200)
    ap.add_argument("--lr",               type=float, default=1e-3)
    ap.add_argument("--weight_decay",     type=float, default=1e-4)
    ap.add_argument("--patience",         type=int,   default=30)
    ap.add_argument("--t0",               type=int,   default=30,
                    help="CosineAnnealingWarmRestarts 的 T_0（第一次重啟週期）")
    ap.add_argument("--num_workers",      type=int,   default=0)
    ap.add_argument("--seed",             type=int,   default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")
    print(f"[Model] 升級版 MelCNN（channels: 64→128→256→256, Hybrid GAP+GMP）")
    print(f"[SpecAugment] {'啟用' if args.augment else '停用'}")
    print(f"[去靜音] top_db={args.top_db}")
    print(f"[LR Scheduler] CosineAnnealingWarmRestarts T_0={args.t0}")

    cache_tag = f"cnn_v4_seg{args.segment_sec}_sr{args.target_sr}_mels{args.n_mels}_frames{args.target_frames}"

    df = pd.read_csv(args.csv_path)
    df.columns = [c.strip() for c in df.columns]
    df["ID"]    = df["ID"].astype(str).str.strip()
    df["set"]   = df["set"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].astype(str).str.strip()

    train_df = df[df["set"] == "train"].copy()
    val_df   = df[df["set"].isin(["val", "dev", "valid"])].copy()

    label2idx, idx2label = build_label_map(train_df)
    num_classes = len(label2idx)
    print(f"[Classes] {num_classes}: {list(label2idx.keys())}")

    with open(os.path.join(args.out_dir, "label_map.json"), "w") as f:
        json.dump({"label2idx": label2idx, "idx2label": idx2label}, f, indent=2)

    # 驗證集（固定切割，可快取）
    print("\n[Step 1] 提取驗證集 Mel Spectrogram...")
    val_songs = []
    for _, row in val_df.iterrows():
        path = os.path.join(args.audio_root, row["ID"])
        if not os.path.isfile(path) or row["label"] not in label2idx:
            continue
        try:
            specs = load_or_extract_val(
                row["ID"], args.audio_root, args.cache_dir, cache_tag,
                args.segment_sec, args.target_sr, args.n_mels,
                args.n_fft, args.hop_length, args.target_frames,
            )
            val_songs.append((row["ID"], label2idx[row["label"]], specs))
        except Exception as e:
            print(f"  [ERROR] {row['ID']}: {e}")
    print(f"  驗證歌曲數: {len(val_songs)}")

    # 訓練集
    print("\n[Step 2] 建立訓練集 Dataset（預載 waveform + 去靜音）...")
    train_dataset = MultiCropDataset(
        train_df, args.audio_root, label2idx,
        args.segment_sec, args.target_sr, args.n_mels, args.n_fft,
        args.hop_length, args.target_frames,
        samples_per_song = args.samples_per_song,
        top_db           = args.top_db,
        augment          = args.augment,
        freq_mask_param  = args.freq_mask_param,
        time_mask_param  = args.time_mask_param,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"  有效訓練樣本數: {len(train_dataset)} ({len(train_df)} 首 × {args.samples_per_song} 片/epoch)")

    # 模型
    model = MelCNN(num_classes=num_classes).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[Model] MelCNN v4 | 參數量: {total_params:,}")

    loss_fn   = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # CosineAnnealingWarmRestarts：T_0 epoch 後重啟，T_mult=2 讓每次週期加倍
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.t0, T_mult=2, eta_min=1e-5
    )

    print(f"\n[Step 3] 開始訓練（最多 {args.epochs} epochs，patience={args.patience}）...")

    best_val_acc   = -1.0
    patience_count = 0
    best_path      = os.path.join(args.out_dir, "checkpoint_best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss   = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            total_loss    += loss.item() * len(y)
            total_correct += (logits.argmax(1) == y).sum().item()
            total_samples += len(y)

        scheduler.step()

        train_loss = total_loss / total_samples
        train_acc  = total_correct / total_samples * 100

        model.eval()
        correct = 0
        with torch.no_grad():
            for _, true_label, specs in val_songs:
                x = torch.from_numpy(specs).unsqueeze(1).to(device)
                preds  = model(x).argmax(1).cpu().numpy()
                final  = int(np.bincount(preds).argmax())
                correct += int(final == true_label)
        val_acc = correct / len(val_songs) * 100

        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            patience_count = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "label2idx": label2idx,
                "idx2label": idx2label,
                "val_acc":   best_val_acc,
                "epoch":     epoch,
                "args":      vars(args),
            }, best_path)
            flag = " ← BEST"
        else:
            patience_count += 1
            flag = ""

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d} | loss={train_loss:.4f} | train={train_acc:.1f}% | val={val_acc:.1f}% | lr={lr_now:.2e}{flag}")

        if patience_count >= args.patience:
            print(f"\n[Early Stop] {args.patience} epochs 無進步，停止訓練。")
            break

    print(f"\n[Done] Best val acc: {best_val_acc:.1f}% | 儲存至 {best_path}")


if __name__ == "__main__":
    main()
