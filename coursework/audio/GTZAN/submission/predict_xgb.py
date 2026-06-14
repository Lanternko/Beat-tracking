#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
路線 B：XGBoost 推論腳本
=====================================================
讀取 gtzan.csv 中 set=test 的欄位，對每首歌曲：
  1. 切割成多個片段（與訓練時相同參數）
  2. 提取特徵 → 標準化
  3. XGBoost 各片段預測 → 多數決 → 最終類別

指令範例：
  python predict.py \\
    --csv_path gtzan.csv \\
    --audio_root ./genres \\
    --ckpt_path ./checkpoint_xgb/checkpoint_best.pt \\
    --out_csv submission.csv
"""

import os
import pickle
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import librosa


# ============================================================
# 特徵提取（與 train.py 完全一致）
# ============================================================
def extract_segment_features(y: np.ndarray, sr: int, n_mfcc: int = 40) -> np.ndarray:
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


def extract_song_features(
    file_path: str,
    segment_sec: float = 3.0,
    target_sr: int = 22050,
    n_mfcc: int = 40,
) -> np.ndarray:
    y, sr   = librosa.load(file_path, sr=target_sr, mono=True)
    seg_len = int(segment_sec * sr)
    n_seg   = len(y) // seg_len

    feats = []
    for i in range(max(n_seg, 1)):
        seg = y[i*seg_len : (i+1)*seg_len] if n_seg > 0 else y
        feats.append(extract_segment_features(seg, sr, n_mfcc=n_mfcc))
    return np.array(feats)


# ============================================================
# 主推論流程
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path",   type=str, required=True,  help="gtzan.csv 路徑")
    ap.add_argument("--audio_root", type=str, required=True,  help="音訊檔案根目錄")
    ap.add_argument("--ckpt_path",  type=str, required=True,  help="checkpoint_best.pt 路徑")
    ap.add_argument("--out_csv",    type=str, default="submission.csv")
    # 以下參數若不指定，自動從 checkpoint 讀取
    ap.add_argument("--segment_sec", type=float, default=None)
    ap.add_argument("--target_sr",   type=int,   default=None)
    ap.add_argument("--n_mfcc",      type=int,   default=None)
    args = ap.parse_args()

    # --- 載入模型 ---
    with open(args.ckpt_path, "rb") as f:
        ckpt = pickle.load(f)

    model     = ckpt["model"]
    scaler    = ckpt["scaler"]
    idx2label = {int(k): v for k, v in ckpt["idx2label"].items()}

    # 優先使用命令列參數，否則從 checkpoint 讀取訓練時的設定
    segment_sec = args.segment_sec or ckpt.get("segment_sec", 3.0)
    target_sr   = args.target_sr   or ckpt.get("target_sr",   22050)
    n_mfcc      = args.n_mfcc      or ckpt.get("n_mfcc",      40)

    print(f"[Config] segment_sec={segment_sec}, target_sr={target_sr}, n_mfcc={n_mfcc}")

    # --- 讀取測試集 CSV ---
    df = pd.read_csv(args.csv_path)
    df.columns = [c.strip() for c in df.columns]
    df["ID"]  = df["ID"].astype(str).str.strip()
    df["set"] = df["set"].astype(str).str.strip().str.lower()

    test_df = df[df["set"] == "test"].copy()
    if len(test_df) == 0:
        raise ValueError("找不到 set=test 的資料，請確認 csv_path 是否為 gtzan.csv")

    print(f"[Test] 共 {len(test_df)} 首歌曲需要預測")

    # --- 推論 ---
    results = []
    for _, row in test_df.iterrows():
        audio_id = row["ID"]
        path     = os.path.join(args.audio_root, audio_id)

        if not os.path.isfile(path):
            print(f"  [WARN] 找不到檔案：{path}，使用預設類別 0")
            results.append((audio_id, idx2label[0]))
            continue

        try:
            segs        = extract_song_features(path, segment_sec, target_sr, n_mfcc)
            segs_scaled = scaler.transform(segs)
            seg_preds   = model.predict(segs_scaled)
            # 多數決：取各片段預測中票數最多的類別
            final_idx   = int(np.bincount(seg_preds).argmax())
            final_label = idx2label[final_idx]
        except Exception as e:
            print(f"  [ERROR] {audio_id}: {e}，使用預設類別 0")
            final_label = idx2label[0]

        results.append((audio_id, final_label))
        print(f"  {audio_id} -> {final_label}")

    # --- 輸出 CSV ---
    out_df = pd.DataFrame(results, columns=["ID", "label"])
    out_df.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"\n[Done] 已儲存預測結果：{args.out_csv}  ({len(out_df)} 筆)")


if __name__ == "__main__":
    main()