# 音樂流派分類實驗記錄
**課程：114-2 Neural Network, NTNU**
**截止日：2026/03/29**
**Kaggle Username：61447101s**

---

## 實驗總覽

| 版本 | 架構 | 訓練策略 | Val Acc | Kaggle Public |
|------|------|----------|---------|---------------|
| Baseline LSTM | 2-layer LSTM | MFCC+特徵, 固定前3秒 | ~70% | — |
| XGBoost | XGBoost | MFCC+Delta+Chroma+Contrast+Tonnetz+ZCR+RMS, 10段多數決 | 86% | — |
| CNN v1（失敗）| MelCNN (32→64→128→128) | Mel Spec, 固定前3秒, 訓練/推論不匹配 | — | — |
| CNN v3 | MelCNN (32→64→128→128) | Mel Spectrogram, 固定前3秒 | 91% | 1.000 |
| TE on v3（退步）| 同 v3 | Temporal Ensemble 10段，推論時看中後段 | 90% | — |
| CNN v2 系列（失敗）| 同 v3 | Random Crop，每 epoch 每首只取 1 段 | 72% | — |
| CNN v3 Random Crop（欠擬合）| 同 v3 | Random Crop 10段/epoch，模型容量不足 | 72% | — |
| CNN v4 | MelCNN (64→128→256→256) + Hybrid Pool | Random Crop + 去靜音 + CosineWarmRestart | 94% | 1.000 |
| CNN v4_long | 同 v4 | patience=50, epochs=300 | 94% | — |
| CNN v5_augment | 同 v4 | + Mild SpecAugment (p=0.5, F=10, T=10) | 94% | — |
| Soft Voting Ensemble | v4_long + v5_augment | Temporal Ensemble × 2 模型平均 | 94% | **1.000** ✅ |

---

## 實驗零：XGBoost Baseline（Route B）

### 動機

在開始訓練 CNN 之前，先以 XGBoost 建立一個快速 Baseline，評估純統計特徵（不含時序資訊）的上限。

### 特徵設計

每首 30 秒音訊切成 10 個 3 秒片段，每段提取以下特徵，各取 mean + std：

| 特徵 | 維度 | 說明 |
|------|------|------|
| MFCC (n=40) | 80 | 音色靜態特徵 |
| Delta-MFCC | 80 | 音色動態變化 |
| Chroma | 24 | 和聲與音高結構 |
| Spectral Contrast | 14 | 頻譜峰谷差異 |
| Tonnetz | 12 | 和聲關係拓撲 |
| ZCR | 2 | 訊號雜訊程度 |
| RMS Energy | 2 | 音量能量 |
| **合計** | **214** | |

推論時對 10 個片段分別預測，取多數決（majority voting）作為最終類別。

### 超參數調整過程

| max_depth | n_estimators | lr | subsample | colsample | Val Acc |
|-----------|-------------|-----|-----------|-----------|---------|
| 6 | 500 | 0.05 | 0.8 | 0.8 | 84% |
| 4 | 3000 (early stop) | 0.03 | 0.7 | 0.7 | **86%** ✅ |
| 3 | 1000 | 0.02 | 0.7 | 0.7 | 86% |
| 4 | 500 (5秒片段) | 0.05 | 0.8 | 0.8 | 84% |

### 訓練指令

```bash
python train_xgb.py \
  --csv_path gtzan.csv \
  --audio_root ./genres \
  --out_dir ./checkpoint_xgb \
  --cache_dir ./feature_cache \
  --max_depth 4 \
  --n_estimators 3000 \
  --learning_rate 0.03 \
  --subsample 0.7 \
  --colsample 0.7
```

### 結論

**Best Val Acc = 86%**，確認 XGBoost 天花板約在此，原因是 mean/std 統計特徵丟失了長期時序資訊，而這正是區分流派的關鍵。決定以 CNN + Mel Spectrogram 作為主要路線。

---

## 實驗一（前置）：CNN v1 失敗 — 訓練推論架構不匹配

### 背景

在 CNN v3 之前，曾嘗試直接對舊版 `predict_cnn.py` 套用新訓練出的 CNN 模型，結果出現以下錯誤：

```
RuntimeError: size mismatch for block1.0.weight:
  copying a param with shape torch.Size([64, 1, 3, 3])
  the shape in current model is torch.Size([32, 1, 3, 3])
```

### 根本原因

`predict_cnn.py` 中的 `MelCNN` 類別定義與 `train_cnn.py` 中的版本不一致——訓練腳本已升級架構（channels 翻倍），但推論腳本未同步更新，導致 `load_state_dict()` 時維度不匹配。

### 修正

每次修改 `train_cnn.py` 的模型架構後，必須同步更新 `predict_cnn.py` 中的 `MelCNN` 定義，兩者必須完全一致。

**教訓：模型架構修改時，訓練腳本和推論腳本要同步維護，否則會產生 silent failure 或直接報錯。**

---

## 實驗二：Temporal Ensemble 初試 — 套用在 CNN v3 反而退步

### 動機

CNN v3 val acc 達到 91%，Kaggle Public Score 1.000（滿分）。參考 NotebookLM 文獻建議，嘗試加入 **Temporal Ensemble**：推論時將 30 秒音訊均勻切 10 段，各段 softmax 機率取平均後再 argmax，理論上能平滑極端片段的雜訊。

### 結果

```
Val Accuracy with Temporal Ensemble: 90/100 = 90.00%
```

**比單一預測的 91% 還退步了 1%。**

### 失敗原因分析

這是一個典型的**訓練與推論策略不匹配（Train-Inference Mismatch）**問題：

```
訓練時：模型只看過每首歌的「前 3 秒」（前奏）
推論時：Temporal Ensemble 強行把中段、後段（主歌、副歌、Solo）也丟給模型預測
```

模型從未在訓練時見過歌曲中後段的頻譜分佈，這些「陌生片段」對模型來說是雜訊，反而干擾了原本正確的預測。

**結論：Temporal Ensemble 的前提是訓練時也要用多段取樣（Random Crop），讓模型見過整首歌的各個部分。**

---

## 實驗三：CNN v2 系列 — Random Crop 引發欠擬合

### 動機

根據實驗二的失敗原因，決定修正訓練策略：**訓練時也改用隨機取段（Random Crop）**，讓模型見過整首歌，再搭配 Temporal Ensemble 推論。

### v2a：SpecAugment 過強

第一版同時加入 Random Crop + SpecAugment（freq_mask=15, time_mask=15），結果 val acc 只有 72%。

懷疑是 SpecAugment 遮蔽太強，先關掉排查。

### v2b：關掉 SpecAugment，還是 72%

關掉 SpecAugment，只保留 Random Crop，val acc 仍然只有 72%。

### 根本問題診斷

| 指標 | v3（固定前3秒）| v2b（Random Crop）|
|------|--------------|-----------------|
| Train Acc | ~95%+ | **52%** |
| Val Acc | 91% | 72% |

Train Acc 卡在 52% 是關鍵訊號，代表**模型容量不足（Underfitting）**：

- v2b 每首歌每 epoch 只取 **1 個**隨機片段 → 有效訓練樣本只有 800 筆
- 但 Random Crop 讓同一首歌的不同片段（前奏、主歌、副歌）頻譜差異極大，等同放大了 10 倍以上的 intra-class variance
- 50 萬參數的小模型無法映射如此高變異性的輸入到正確標籤

### v2c：修正樣本數不足（MultiCropDataset）

將 Dataset 長度膨脹為 `歌曲數 × samples_per_song`（預設 10），每次取不同隨機位置，確保有效訓練樣本 = 8000，與原版相同。

結果：train acc 仍在 52% 左右，val acc 72%。

**確認問題不在樣本數，而是 50 萬參數的模型容量根本不足以應對 Random Crop 帶來的高內部變異性。**

---

## 實驗四：CNN v3 → CNN v4（核心突破）

### 問題背景

CNN v3 在 val set 達到 91%，但訓練策略存在根本缺陷：**模型只看每首歌的前 3 秒**（`timeseries_length=128, hop_length=512, sr=22050` → 約 2.97 秒）。這導致模型可能記住的是前奏的底噪或特定樂器音色，而非整首歌的流派特徵。

當改為隨機取段（Random Crop）後，train acc 卡在 52%，確認了 v3 的 50 萬參數容量不足以應對整首歌各段落的高內部變異性。

### 升級內容

**1. 模型加深加寬（Channels 全面翻倍）**

```
v3: Conv(32) → Conv(64)  → Conv(128) → Conv(128)  參數量 ~50萬
v4: Conv(64) → Conv(128) → Conv(256) → Conv(256)  參數量 ~110萬
```

**2. Hybrid Pooling（取代 Flatten + LazyLinear）**

```python
# v3：直接 Flatten，丟失空間彈性
self.classifier = nn.Sequential(
    nn.Flatten(),
    nn.LazyLinear(128),  # 依賴輸入尺寸自動推斷
    ...
)

# v4：GAP + GMP 並聯，各保留不同層面的特徵
self.gap = nn.AdaptiveAvgPool2d(1)  # 全局平均：保留分佈
self.gmp = nn.AdaptiveMaxPool2d(1)  # 全局最大：捕捉突出特徵
# concat → Linear(512→256) → Linear(256→10)
```

文獻支持：Max Pooling 捕捉突出局部特徵，Average Pooling 保留全局分佈，兩者互補。

**3. 去靜音前處理**

```python
y_trimmed, _ = librosa.effects.trim(y, top_db=20)
```

避免隨機取段時取到無聲或靜音片段，確保每個訓練樣本都含有效音樂內容。

**4. 學習率排程：CosineAnnealingWarmRestarts**

```
v3：ReduceLROnPlateau（被動等 plateau 才降 LR）
v4：CosineAnnealingWarmRestarts T_0=30（主動週期性重啟）
```

週期性重啟讓模型能跳出局部最優，在訓練後期仍持續找到更好的解。

### 訓練指令

```bash
python train_cnn.py \
  --csv_path gtzan.csv \
  --audio_root ./genres \
  --out_dir ./checkpoint_cnn_v4 \
  --cache_dir ./feature_cache_cnn_v4 \
  --epochs 200 \
  --patience 30 \
  --samples_per_song 10
```

### 訓練曲線（重要節點）

| Epoch | Train Acc | Val Acc | 備註 |
|-------|-----------|---------|------|
| 1 | 32.0% | 48.0% | 起點 |
| 8 | 61.2% | 84.0% | 快速上升 |
| 20 | 76.3% | 88.0% | ← BEST 當時 |
| 36 | 77.4% | 90.0% | 重啟後突破 |
| 43 | 82.0% | 91.0% | |
| 49 | 85.2% | 93.0% | ← BEST 當時 |
| 67 | 91.2% | 94.0% | ← BEST 最終 |
| 97 | 89.1% | 93.0% | Early Stop（patience=30 耗盡）|

**最終結果：Best Val Acc = 94.0%（epoch 67）**

### 推論：Temporal Ensemble

訓練完成後，改用 Temporal Ensemble 推論策略：

```
傳統推論：只取前 3 秒 → 單一預測
Temporal Ensemble：將 30 秒音訊均勻切 10 段 → 各段 softmax 機率取平均 → argmax
```

```bash
python predict_cnn.py \
  --csv_path gtzan.csv \
  --audio_root ./genres \
  --ckpt_path ./checkpoint_cnn_v4/checkpoint_best.pt \
  --out_csv val_pred_v4.csv \
  --predict_set val \
  --n_segments 10
```

**Val Accuracy with Temporal Ensemble：92/100 = 92.00%**

> 注意：訓練時的 val 評估用 majority vote（多段投票），推論時用 softmax averaging，兩者邏輯略有差異，導致結果為 92% 而非 94%。

**Kaggle Public Score：1.000（20/20）**

---

## 實驗四補充：predict_cnn.py 架構不同步 Bug

### 問題

CNN v4 訓練完成後，執行推論時出現 `RuntimeError`：

```
size mismatch for block1.0.weight:
  copying a param with shape torch.Size([64, 1, 3, 3])
  the shape in current model is torch.Size([32, 1, 3, 3])
```

### 根本原因

`predict_cnn.py` 裡的 `MelCNN` 定義仍是舊版架構（channels: 32→64→128→128），但 `checkpoint_cnn_v4` 是用新版 `train_cnn.py`（channels: 64→128→256→256 + Hybrid Pooling）訓練的，兩個檔案未同步。

### 修正

將 `predict_cnn.py` 中的 `MelCNN` class 更新為 v4 架構，加入 `self.gap`、`self.gmp`，並將 classifier 改為 `Linear(512→256→10)`。

**教訓：每次升級 `train_cnn.py` 的模型架構，必須同步更新 `predict_cnn.py`。**

### 修正後推論結果

```bash
python predict_cnn.py \
  --csv_path gtzan.csv \
  --audio_root ./genres \
  --ckpt_path ./checkpoint_cnn_v4/checkpoint_best.pt \
  --out_csv val_pred_v4.csv \
  --predict_set val \
  --n_segments 10
```

**Val Accuracy（Temporal Ensemble）：92/100 = 92.00%**
**Kaggle Public Score：1.000（20/20）✅**

> 訓練時 val 評估用 majority vote，推論時用 softmax averaging，邏輯略有差異，導致結果為 92% 而非訓練時看到的 94%。

---

## 實驗五：CNN v4_long（拉長 Patience）

### 訓練速度危機：I/O Bound 瓶頸

實驗五與六啟動後，發現每個 epoch 需要 **5-6 分鐘**，300 epochs 預估需要 **30 小時**，無法在截止日前完成。

**根本原因：`num_workers=0`**

DataLoader 預設 `num_workers=0`，所有 Mel Spectrogram 計算都在主線程 on-the-fly 進行，GPU 大部分時間在等 CPU 準備資料（I/O Bound），實際 GPU 使用率極低。

**解法：加入 `--num_workers 4`**

兩個實驗先停止，加上 `--num_workers 4` 重跑：

```bash
tmux kill-session -t cnn_v4_long
tmux kill-session -t cnn_v5_augment
```

DataLoader 改為 4 個子進程並行載入，CPU 的資料準備不再阻塞 GPU 計算。

- **修正前**：每 epoch ~5-6 分鐘，300 epochs = ~30 小時
- **修正後**：每 epoch < 1 秒，300 epochs = ~3 分鐘（**加速 300 倍以上**）

> **關鍵學習**：Deep learning 訓練的瓶頸常在資料載入（I/O Bound），而非模型計算。遇到 GPU 使用率低但訓練慢的情況，優先檢查 `num_workers` 設定。

### 訓練指令

```bash
python train_cnn.py \
  --csv_path gtzan.csv \
  --audio_root ./genres \
  --out_dir ./checkpoint_cnn_v4_long \
  --cache_dir ./feature_cache_cnn_v4 \
  --epochs 300 \
  --patience 50 \
  --samples_per_song 10 \
  --num_workers 4
```

> **加速說明**：加入 `--num_workers 4` 後，DataLoader 改為多進程並行載入，每 epoch 從原本 5-6 分鐘壓縮到 ~1 秒以內，300 epochs 從預估 30 小時壓到約 3 分鐘。這是解決 I/O Bound 瓶頸的標準作法。

### 結果

| 指標 | 數值 |
|------|------|
| Best Val Acc | **94.0%** |
| 達到 BEST 的 Epoch | 58 |
| Early Stop Epoch | ~108 |

**結論：拉長 patience 並未突破 94% 天花板。**

---

## 實驗六：CNN v5_augment（Mild SpecAugment）

### 動機

SpecAugment 是音訊領域常用的資料增強方法，透過隨機遮蔽頻率軸與時間軸，強迫模型不依賴局部頻率噪音或單一瞬間特徵，提升對未見過資料的泛化能力（Robustness）。

目標是改善 Private Test（80筆）的泛化能力，而非 val accuracy。

### 參數設定依據

根據文獻中針對音樂流派分類的資料增強建議，遮蔽比例應控制在 5%~15%：

```
Mel Spectrogram 尺寸：(128 mels, 128 time frames)
5% ~ 15% 換算：param = 6 ~ 19
最終選擇：freq_mask_param=10, time_mask_param=10（落在甜蜜點）
觸發機率：p=0.5（50% 機率觸發，確保模型仍能學到完整特徵）
```

### 程式碼修改

**`train_cnn.py` 中的 `spec_augment()` 函式加入機率控制：**

```python
def spec_augment(mel, freq_mask_param=10, time_mask_param=10, p=0.5):
    if random.random() > p:
        return mel  # 50% 機率直接跳過
    # ... 原本的遮蔽邏輯
```

### 訓練指令

```bash
python train_cnn.py \
  --csv_path gtzan.csv \
  --audio_root ./genres \
  --out_dir ./checkpoint_cnn_v5_augment \
  --cache_dir ./feature_cache_cnn_v4 \
  --epochs 300 \
  --patience 50 \
  --samples_per_song 10 \
  --num_workers 4 \
  --augment \
  --freq_mask_param 10 \
  --time_mask_param 10
```

### 結果

| 指標 | 數值 |
|------|------|
| Best Val Acc | **94.0%** |
| 達到 BEST 的 Epoch | 48（比 v4_long 的 58 更快）|
| Early Stop Epoch | ~98 |

**觀察：SpecAugment 讓模型更快收斂（epoch 48 vs 58），但最終 val acc 相同。**

---

## 實驗七：Soft Voting Ensemble

### 動機

單一模型可能存在決策盲區，兩個模型在犯錯樣本上若有差異，就能透過機率平均互補，這是 Kaggle 競賽中常見的「榨出最後 1-2%」策略。

### 實作方式

```
對每首歌：
  模型 A（v4_long）   × 10 段 → softmax 機率
  模型 B（v5_augment）× 10 段 → softmax 機率
  平均所有機率 → argmax → 最終預測

等效於：avg_probs = (probs_A + probs_B) / 2
```

### 推論指令

```bash
# Val 驗證
python predict_ensemble.py \
  --csv_path gtzan.csv \
  --audio_root ./genres \
  --ckpt_paths ./checkpoint_cnn_v4_long/checkpoint_best.pt \
              ./checkpoint_cnn_v5_augment/checkpoint_best.pt \
  --out_csv val_ensemble.csv \
  --predict_set val \
  --n_segments 10

# Test 提交
python predict_ensemble.py \
  --csv_path gtzan.csv \
  --audio_root ./genres \
  --ckpt_paths ./checkpoint_cnn_v4_long/checkpoint_best.pt \
              ./checkpoint_cnn_v5_augment/checkpoint_best.pt \
  --out_csv submission_ensemble.csv \
  --n_segments 10
```

### 結果

**Val Accuracy：94.00%（與單模型相同，無提升）**

### 失敗原因分析

Ensemble 無效的核心原因是**模型多樣性不足**（Lack of Model Diversity）：

- 兩個模型架構完全相同（MelCNN v4）
- 訓練資料完全相同（同一份 GTZAN 800 首）
- 唯一差異是 SpecAugment 的有無，影響過小

**6 個錯誤樣本完全相同**，代表兩個模型的決策邊界高度重疊，Soft Voting 無法互補。

### 錯誤分析（6 個共同錯誤）

| 音訊 ID | 真實標籤 | 預測標籤 | 可能原因 |
|---------|----------|----------|----------|
| 00930.au | rock | disco | Disco 有強節拍與電吉他，與 rock 音色相近 |
| 00931.au | country | reggae | 兩者都有輕鬆節奏感與相似 texture |
| 00698.au | hiphop | disco | 強節拍特徵重疊 |
| 00112.au | hiphop | rock | 電吉他採樣常見於 hiphop |
| 00048.au | reggae | rock | Reggae 電吉他成分被誤判 |
| 00597.au | pop | hiphop | 現代 pop 大量借用 hiphop 製作風格 |

這些誤分類在音樂上都有邏輯，體現了**流派模糊性（Genre Ambiguity）**的本質——這是 GTZAN 資料集「玻璃天花板」的具體呈現。

---

## 最終提交

| 提交 | 檔案 | Val Acc | Kaggle Public |
|------|------|---------|---------------|
| 第一次 | `submission_cnn_v4.csv` | 92%（Temporal Ensemble） | **1.000** ✅ |
| 最終版 | `submission_ensemble.csv` | 94%（Soft Voting Ensemble） | **1.000** ✅ |

兩次提交均達到 Public Score 滿分。Val Accuracy 從 92% 提升至 94% 代表推論策略改善，而非資訊洩漏——兩者用的都是嚴格分隔的 val set。

**GTZAN 理論極限（根據 Sturm 的缺陷分析）：~94.5%**
本專案的 94% 已觸及資料集的有效分類上限，繼續訓練不會帶來有意義的提升。

---

## 關鍵學習

1. **I/O Bound 瓶頸**：`num_workers=0` 讓每 epoch 需要 5-6 分鐘；改為 `num_workers=4` 壓到 1 秒以內，加速 300 倍以上。Deep learning 訓練的瓶頸常在資料載入，而非模型計算。遇到 GPU 使用率低但訓練慢的情況，優先檢查 DataLoader 的 `num_workers` 設定。

2. **訓練推論策略必須一致（Train-Inference Consistency）**：Temporal Ensemble 套用在只看前 3 秒訓練的模型上反而退步，因為模型從未見過歌曲中後段。訓練時用 Random Crop，推論時才能用 Temporal Ensemble。

3. **模型容量與 Random Crop 的匹配**：小模型（50 萬參數）遇到 Random Crop 的高變異性會欠擬合；加大模型（110 萬參數）才能真正學習整首歌的特徵。擴大容量是正確解，而非繼續加強資料增強。

4. **Ensemble 的前提是多樣性**：同架構、同資料的模型做 Ensemble 效果有限——本專案 v4_long 與 v5_augment 的 6 個錯誤樣本完全相同，Soft Voting 無法互補。真正有效的 Ensemble 需要不同架構（CNN + Transformer）或不同特徵（Mel + MFCC）。

5. **94% 是 GTZAN 的有效天花板**：根據 Bob L. Sturm 的研究，GTZAN 存在錯誤標籤與重複音軌，完美分類器的理論極限約為 94.5%。超過此數字代表模型在死背資料集雜訊，而非真正學習音樂流派。本專案的 94% 已達到有意義的上限。
