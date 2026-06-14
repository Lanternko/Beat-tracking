#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
路線 B：XGBoost + 豐富音訊特徵 音樂流派分類器（含特徵快取）
=====================================================
核心設計：
  1. 將每首 30 秒音訊切成多個不重疊片段（預設 3 秒）
     → 以整首歌為單位切分，防止資料外洩
  2. 每個片段提取豐富特徵：
     MFCC(40) + Delta-MFCC(40) + Chroma(12) + Contrast(7) + Tonnetz(6) + ZCR(1) + RMS(1)
     每種特徵取 mean + std → 展平成一維向量
  3. 特徵快取：第一次提取後存成 .npz，之後直接讀取省時間
  4. XGBoost 分類器訓練
  5. 驗證時：對每首歌所有片段分別預測，取多數決（majority voting）

輸出：
  - checkpoint_best.pt（pickle 格式，含模型 + scaler + label mapping）
  - label_map.json
  - feature_cache/（快取的特徵檔）

指令範例（第一次跑，建立快取）：
  python train_xgb.py \\
    --csv_path gtzan.csv \\
    --audio_root ./genres \\
    --out_dir ./checkpoint_xgb \\
    --cache_dir ./feature_cache

之後調整超參數，直接讀快取（秒速完成特徵提取）：
  python train_xgb.py \\
    --csv_path gtzan.csv \\
    --audio_root ./genres \\
    --out_dir ./checkpoint_xgb \\
    --cache_dir ./feature_cache \\
    --max_depth 4 --n_estimators 800
"""

import os
import json
import pickle
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import librosa

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb


# ============================================================
# 1) 特徵提取：單一片段 -> 特徵向量
# ============================================================
def extract_segment_features(y: np.ndarray, sr: int, n_mfcc: int = 40) -> np.ndarray:
    """
    從一段音訊波形提取豐富特徵向量。

    特徵組合（文獻支持）：
      - MFCC (n_mfcc 維) + Delta-MFCC (n_mfcc 維)：靜態 + 動態音色
      - Chroma (12 維)：和聲與音高結構
      - Spectral Contrast (7 維)：頻譜峰谷差異
      - Tonnetz (6 維)：和聲關係拓撲特徵
      - ZCR (1 維)：訊號噪音程度
      - RMS Energy (1 維)：音量能量

    每種特徵取 mean + std（沿時間軸）
    預設 n_mfcc=40 → 特徵維度 = (40*2 + 12+7+6+1+1) * 2 = 214 維
    """
    hop_length = 512

    mfcc     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    delta    = librosa.feature.delta(mfcc)
    chroma   = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)
    tonnetz  = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    zcr      = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)
    rms      = librosa.feature.rms(y=y, hop_length=hop_length)

    def stats(feat):
        return np.concatenate([feat.mean(axis=1), feat.std(axis=1)])

    return np.concatenate([
        stats(mfcc), stats(delta), stats(chroma),
        stats(contrast), stats(tonnetz), stats(zcr), stats(rms),
    ]).astype(np.float32)


# ============================================================
# 2) 音訊切割：整首歌 -> 多個片段特徵矩陣
# ============================================================
def extract_song_features(
    file_path: str,
    segment_sec: float = 3.0,
    target_sr: int = 22050,
    n_mfcc: int = 40,
) -> np.ndarray:
    """
    將一首歌切成多個不重疊片段，回傳 shape: (n_segments, feature_dim)
    """
    y, sr   = librosa.load(file_path, sr=target_sr, mono=True)
    seg_len = int(segment_sec * sr)
    n_seg   = len(y) // seg_len

    feats = []
    for i in range(max(n_seg, 1)):
        seg = y[i*seg_len : (i+1)*seg_len] if n_seg > 0 else y
        feats.append(extract_segment_features(seg, sr, n_mfcc=n_mfcc))
    return np.array(feats)


# ============================================================
# 3) 快取工具函式
# ============================================================
def get_cache_path(cache_dir: str, audio_id: str, cache_tag: str) -> str:
    """將音訊 ID 轉成快取檔路徑，並以 cache_tag 區分不同參數"""
    safe_id = audio_id.replace("/", "_").replace("\\", "_")
    return os.path.join(cache_dir, cache_tag, f"{safe_id}.npz")


def load_or_extract(
    audio_id: str,
    audio_root: str,
    cache_dir: str,
    cache_tag: str,
    segment_sec: float,
    target_sr: int,
    n_mfcc: int,
) -> np.ndarray:
    """
    若快取存在則直接讀取，否則提取特徵並存檔。
    回傳 shape: (n_segments, feature_dim)
    """
    path = os.path.join(audio_root, audio_id)

    if cache_dir:
        cache_path = get_cache_path(cache_dir, audio_id, cache_tag)
        if os.path.isfile(cache_path):
            return np.load(cache_path)["feats"]

    # 提取特徵
    feats = extract_song_features(path, segment_sec, target_sr, n_mfcc)

    # 存快取
    if cache_dir:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, feats=feats)

    return feats


# ============================================================
# 4) Label 映射
# ============================================================
def build_label_map(df: pd.DataFrame):
    labels    = sorted({str(x).strip() for x in df["label"].dropna()})
    label2idx = {lab: i for i, lab in enumerate(labels)}
    idx2label = {i: lab for lab, i in label2idx.items()}
    return label2idx, idx2label


# ============================================================
# 5) 主訓練流程
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path",      type=str,   required=True)
    ap.add_argument("--audio_root",    type=str,   required=True)
    ap.add_argument("--out_dir",       type=str,   default="checkpoint_xgb")
    ap.add_argument("--cache_dir",     type=str,   default=None,
                    help="快取目錄：第一次提取後存檔，之後直接讀取。建議設為 ./feature_cache")
    ap.add_argument("--segment_sec",   type=float, default=3.0)
    ap.add_argument("--target_sr",     type=int,   default=22050)
    ap.add_argument("--n_mfcc",        type=int,   default=40)
    ap.add_argument("--n_estimators",  type=int,   default=500)
    ap.add_argument("--max_depth",     type=int,   default=6)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--subsample",     type=float, default=0.8)
    ap.add_argument("--colsample",     type=float, default=0.8)
    ap.add_argument("--seed",          type=int,   default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)

    # 快取 tag：包含所有影響特徵的參數，確保不同參數不共用快取
    cache_tag = f"seg{args.segment_sec}_sr{args.target_sr}_mfcc{args.n_mfcc}"
    if args.cache_dir:
        print(f"[Cache] 使用快取目錄：{args.cache_dir}/{cache_tag}/")

    # --- 讀 CSV ---
    df = pd.read_csv(args.csv_path)
    df.columns = [c.strip() for c in df.columns]
    df["ID"]    = df["ID"].astype(str).str.strip()
    df["set"]   = df["set"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].astype(str).str.strip()

    train_df = df[df["set"] == "train"].copy()
    val_df   = df[df["set"].isin(["val", "dev", "valid"])].copy()

    label2idx, idx2label = build_label_map(train_df)
    print(f"[Classes] {len(label2idx)}: {list(label2idx.keys())}")

    with open(os.path.join(args.out_dir, "label_map.json"), "w") as f:
        json.dump({"label2idx": label2idx, "idx2label": idx2label}, f, indent=2)

    # --- 訓練集特徵提取（含快取）---
    print("\n[Step 1] 提取訓練集特徵...")
    X_train, y_train = [], []
    for i, (_, row) in enumerate(train_df.iterrows()):
        path = os.path.join(args.audio_root, row["ID"])
        if not os.path.isfile(path):
            print(f"  [WARN] 找不到：{path}")
            continue
        try:
            segs = load_or_extract(
                row["ID"], args.audio_root, args.cache_dir, cache_tag,
                args.segment_sec, args.target_sr, args.n_mfcc,
            )
            for feat in segs:
                X_train.append(feat)
                y_train.append(label2idx[row["label"]])
            # 進度顯示
            if (i + 1) % 100 == 0:
                print(f"  已處理 {i+1}/{len(train_df)} 首...")
        except Exception as e:
            print(f"  [ERROR] {row['ID']}: {e}")

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print(f"  片段數: {len(X_train)}, 特徵維度: {X_train.shape[1]}")

    # --- 驗證集特徵提取（含快取）---
    print("\n[Step 2] 提取驗證集特徵...")
    val_songs = []
    for _, row in val_df.iterrows():
        path = os.path.join(args.audio_root, row["ID"])
        if not os.path.isfile(path) or row["label"] not in label2idx:
            continue
        try:
            segs = load_or_extract(
                row["ID"], args.audio_root, args.cache_dir, cache_tag,
                args.segment_sec, args.target_sr, args.n_mfcc,
            )
            val_songs.append((row["ID"], label2idx[row["label"]], segs))
        except Exception as e:
            print(f"  [ERROR] {row['ID']}: {e}")
    print(f"  驗證歌曲數: {len(val_songs)}")

    # --- 標準化 ---
    print("\n[Step 3] 特徵標準化...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # --- 準備驗證集片段（供 eval_set 使用）---
    X_val_flat, y_val_flat = [], []
    for _, true_label, segs in val_songs:
        for feat in segs:
            X_val_flat.append(feat)
            y_val_flat.append(true_label)
    X_val_scaled = scaler.transform(np.array(X_val_flat))
    y_val_flat   = np.array(y_val_flat)

    # --- XGBoost 訓練 ---
    print("\n[Step 4] 訓練 XGBoost...")
    print(f"  設定：n_estimators={args.n_estimators}, max_depth={args.max_depth}, lr={args.learning_rate}")
    model = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample,
        eval_metric="mlogloss",
        early_stopping_rounds=50,
        random_state=args.seed,
        n_jobs=-1,
        verbosity=1,
    )
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_train_scaled, y_train), (X_val_scaled, y_val_flat)],
        verbose=50,
    )
    print(f"  最佳樹數：{model.best_iteration + 1}")
    train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
    print(f"  Train acc (片段級): {train_acc*100:.2f}%")

    # --- 驗證（歌曲級多數決）---
    print("\n[Step 5] 驗證集評估（歌曲級多數決）...")
    correct = 0
    for _, true_label, segs in val_songs:
        preds = model.predict(scaler.transform(segs))
        if int(np.bincount(preds).argmax()) == true_label:
            correct += 1
    val_acc = correct / len(val_songs) if val_songs else 0.0
    print(f"  Val acc (歌曲級): {val_acc*100:.2f}%  ({correct}/{len(val_songs)})")

    # --- 儲存模型 ---
    ckpt = {
        "model": model, "scaler": scaler,
        "label2idx": label2idx, "idx2label": idx2label,
        "val_acc": val_acc, "segment_sec": args.segment_sec,
        "target_sr": args.target_sr, "n_mfcc": args.n_mfcc,
    }
    ckpt_path = os.path.join(args.out_dir, "checkpoint_best.pt")
    with open(ckpt_path, "wb") as f:
        pickle.dump(ckpt, f)
    print(f"\n[Done] Val acc: {val_acc*100:.2f}% | 儲存至 {ckpt_path}")


if __name__ == "__main__":
    main()