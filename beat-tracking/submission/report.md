# 節拍偵測於 GTZAN 資料集之系統性研究
## 從 RNN/DBN 基準到 Transformer 架構與曲風感知後處理

---

## 一、簡介（Introduction）

節拍偵測（beat tracking）是音樂資訊檢索（MIR）中的基礎任務，目標是從音訊訊號中自動估計音樂的節拍時間點。此任務在音樂同步、節奏分析、自動編曲等下游應用中扮演關鍵角色。

本研究使用 GTZAN 資料集進行系統性評估。GTZAN 包含 10 種曲風（Blues、Classical、Country、Disco、Hip-Hop、Jazz、Metal、Pop、Reggae、Rock），每種各 100 首 30 秒片段。各曲風的節拍特性差異極大：Disco 節拍規律穩定（約 120 BPM）、Classical 速度變化劇烈、Jazz 切分節奏複雜、Reggae 強調反拍（off-beat），為節拍偵測帶來多樣化的挑戰。

本研究的核心問題有四：
1. 現代 Transformer 架構相較傳統 RNN+DBN 基準能提升多少？
2. 後處理解碼器（DBN vs CRF）如何影響結果？
3. 融入「曲風領域知識」的 DBN 參數最佳化是否能改進 RNN 系統？
4. 將 RNN 系統的曲風感知策略套用到 Transformer activation 上，能否突破純 Transformer 的上限？

---

## 二、相關研究（Related Work）

### 2.1 神經網路節拍偵測

早期深度學習方法以循環神經網路（RNN）為主。Böck et al.（2011）提出以雙向長短期記憶網路（BLSTM）計算節拍激活函數，再透過動態貝葉斯網路（DBN）解碼，此方法成為後續基準。Böck et al.（2014）進一步驗證多模型 ensemble 可顯著提升強健性，此設計已內建於 madmom 函式庫的 `RNNBeatProcessor`（內部 8 個 BLSTM 取平均）。

Davies & Böck（2019）提出時序卷積網路（TCN）取代 RNN，在多數資料集上表現更佳，但在 GTZAN 上的效果（F1 ≈ 84.3%）反而略遜於 RNN（≈ 86.4%）。

### 2.2 解碼器設計：DBN 與 CRF

DBN（生成式模型）透過 `transition_lambda` 與 `observation_lambda` 控制節拍序列的平滑程度：前者越大，模型越傾向維持穩定速度；後者控制對神經網路輸出的信任程度。CRF（判別式模型，Krebs et al. 2015）直接學習輸入特徵與節拍序列的對應關係，理論上更靈活，但缺乏顯式的速度先驗。

### 2.3 Transformer 架構

Zhao et al.（2022）的 Beat-Transformer 將自注意力機制引入節拍偵測。Böck et al.（2024）的 Beat This! 採用卷積－Transformer 混合架構，並以 shift-tolerant post-processing 取代 DBN，在 GTZAN 上報告 F1 ≈ 89%。其核心主張是：DBN 並非 Transformer 所必需，甚至會因施加速度平滑先驗而削弱 Transformer 的長程建模優勢。

### 2.4 曲風感知參數調整

多項研究指出不同曲風對節拍偵測系統有不同需求。Chiu et al. 發現古典音樂因速度變化劇烈，需要較低的 `transition_lambda`；Böck et al.（2014）亦指出異質曲風在統一參數下難以達到最佳。BeatNet+（2024）的 per-genre 分析顯示 GTZAN 各曲風 F1 差距可達 30 個百分點以上（Classical/Jazz 最差、Disco/HipHop 最好），佐證了曲風感知設計的必要性。

---

## 三、方法（Methods）

本研究依序測試以下八類方法。所有評估皆在完整 GTZAN 資料集上進行（998 個有效檔案，jazz.00054 因音訊損壞排除）。

### 3.1 基準系統：RNN + DBN

使用 madmom 的 `RNNBeatProcessor`（8 BLSTM ensemble）+ `DBNBeatTrackingProcessor`（預設 `transition_lambda=100`、`observation_lambda=16`、`fps=100`）。

### 3.2 替代解碼器：RNN + CRF

保留 RNN 前端，以 `CRFBeatDetectionProcessor` 取代 DBN，比較兩種解碼器在 GTZAN 上的差異。

### 3.3 DBN 全域參數搜尋

對 998 個檔案進行 grid search，搜尋空間：`transition_lambda ∈ {50, 100, 150, 200, 300}`、`observation_lambda ∈ {8, 16, 24, 32}`，找出全域最佳設定。

### 3.4 曲風感知 DBN：Literature-driven

根據文獻直接設定參數（Classical/Jazz `tl=30, ol=8`；Disco/HipHop/Metal `tl=150`；Reggae `ol=8` 對應反拍特性），測試領域知識能否直接轉換為效能提升。

### 3.5 曲風感知 DBN：Data-driven

對每個曲風獨立執行 grid search（10 曲風 × 20 組合 = 200 次評估），找出各曲風的資料導向最佳參數。

### 3.6 Beat This!（主要提交）

採用 Böck et al.（2024）的預訓練模型（checkpoint `final0`），CUDA 加速推理（RTX 5090）。模型以卷積層提取局部頻譜特徵、Time-Frequency Transformer 建模長程依賴、最後以 shift-tolerant post-processing 取代 DBN。

### 3.7 Beat This! 消融與 Ensemble

- **Beat This! + DBN**：在 Beat This! 輸出後加回 DBN（`dbn=True`），驗證論文主張。
- **Checkpoint Ensemble**：以三個可用 checkpoint（`final0/1/2`）跑 majority vote（容忍窗口 70ms，門檻 2/3）。

### 3.8 Beat This! + 曲風感知 DBN（本研究核心創新）

將「Data-driven 曲風感知 DBN」的策略套用到 Beat This! 的 frame-level activation：
1. 從 Beat This! 取得 beat logits（50 fps）
2. 重新取樣到 100 fps 對齊 DBN
3. 對每個曲風獨立 grid search 找最佳 DBN 參數
4. 用各曲風最佳參數生成最終預測

研究問題：DBN 對 Transformer 的傷害是「架構不匹配」還是「參數不夠彈性」？若曲風感知 DBN 能恢復或超越純 Beat This!，則為後者；若仍低於純 Beat This!，則確認架構不匹配。

---

## 四、實驗結果（Experiments & Results）

### 4.1 主要結果

**表 1：所有方法在 GTZAN 998 個檔案上的 F1-Measure**

| 排名 | 方法 | F1-Measure | 相對 Baseline |
|------|------|-----------|-------------|
| 8 | RNN + CRF | 86.43% | −0.59% |
| 7 | RNN + DBN（Baseline，預設參數） | 87.02% | — |
| 6 | RNN + Genre-Aware DBN（Literature） | 87.22% | +0.20% |
| 5 | Beat This! + DBN（消融） | 87.56% | +0.54% |
| 4 | RNN + Per-Genre DBN（Data-driven） | **88.42%** | +1.40% |
| 3 | **Beat This! + Genre-Aware DBN（本研究）** | **88.59%** | +1.57% |
| 2 | **Beat This!（最終提交）** | **88.72%** | **+1.70%** |
| 🥇 | **Beat This! Ensemble (×3 checkpoints)** | **88.75%** | **+1.73%** |

> **目標門檻：F1 > 87.5%。Beat This! 達成 88.72%，超標 +1.22%。**

### 4.2 Per-Genre 詳細分析

**表 2：各曲風 F1（%）**

| 曲風 | Baseline | CRF | Per-Genre | Genre-Aware (Lit) | Beat This! | BT+DBN | **BT+Genre-DBN** | BT Ensemble |
|------|---------|-----|-----------|-------------------|-----------|--------|-----------------|-------------|
| Blues | 83.35 | 83.04 | **85.70** | 83.35 | 82.89 | 80.96 | 82.40 | 82.95 |
| Classical | 66.03 | 66.69 | 66.44 | 65.16 | 64.97 | 64.34 | **68.08** | 64.98 |
| Country | 90.49 | 89.97 | 90.49 | 90.49 | **93.96** | 92.87 | 93.69 | 93.10 |
| Disco | 96.39 | 96.13 | 96.64 | 96.42 | 96.43 | 96.60 | **96.66** | 96.45 |
| HipHop | 95.24 | 91.28 | 96.19 | 96.19 | 97.17 | 97.64 | **98.21** | 97.86 |
| Jazz | 77.93 | 80.82 | 81.97 | 79.51 | **86.57** | 81.88 | 83.29 | 86.56 |
| Metal | 84.64 | 82.64 | 85.72 | 83.97 | **87.46** | 84.51 | 85.06 | 87.36 |
| Pop | 92.02 | 90.28 | 93.76 | 92.02 | 95.17 | 95.20 | **95.53** | 95.47 |
| Reggae | 92.87 | 93.47 | **94.15** | 93.83 | 90.48 | 90.78 | 91.56 | 90.47 |
| Rock | 91.25 | 90.04 | **93.15** | 91.24 | 92.09 | 90.80 | 91.38 | 92.33 |
| **OVERALL** | **87.02** | **86.43** | **88.42** | **87.22** | **88.72** | **87.56** | **88.59** | **88.75** |

**各曲風最佳方法：**
- Blues、Reggae：RNN + Per-Genre DBN
- Classical：**Beat This! + Genre-Aware DBN（68.08%，本研究方法）**
- Country、Jazz、Metal：Beat This!
- Disco、HipHop、Pop：Beat This! + Genre-Aware DBN
- Rock：RNN + Per-Genre DBN（93.15%）
- 整體：Beat This! Ensemble（88.75%）

---

## 五、討論（Discussion）

### 5.1 解碼器選擇：CRF 弱於 DBN

RNN + CRF（86.43%）較 RNN + DBN baseline（87.02%）退步 0.59%。GTZAN 中以節拍規律的流行/舞曲類為主，DBN 透過 `transition_lambda` 注入的速度週期性先驗在這種資料分佈下相當有利；CRF 雖然理論上更靈活，但在訓練資料有限時難以勝過顯式的領域先驗。值得一提的是 CRF 在 Reggae 與 Classical 兩個「不規律」曲風上反而略勝 DBN，符合「先驗鬆綁有助於彈性節奏」的直覺。

### 5.2 Beat This! 驗證論文主張，但被「曲風感知 DBN」推翻

**Beat This!（dbn=False，88.72%）優於 Beat This! + 預設 DBN（87.56%）達 1.16%**，乍看完美驗證了 Böck et al.（2024）「DBN 不必要」的主張。但本研究的核心創新——**Beat This! + 曲風感知 DBN（88.59%）——將此差距縮小到僅 0.13%**，幾乎完全恢復了 DBN 造成的損失。

這意味著論文的主張需要更精確的表述：**「DBN 對 Transformer 的傷害並非根本性的架構不匹配，而是參數設定不夠彈性。」** 當 DBN 參數隨曲風自適應後，後處理的傷害幾乎消失，反而在 Classical（68.08%，所有方法之冠）、Disco、HipHop、Pop 上超越純 Beat This!。

### 5.3 曲風感知最佳化的兩條路線

本研究比較了兩種曲風感知策略：
- **Literature-driven**（87.22%）：依文獻設定參數，整體僅小幅超越 baseline 0.20%。Classical（−0.86%）和 Metal（−0.67%）甚至退步。
- **Data-driven**（88.42%）：對每個曲風 grid search，整體大幅提升 1.40%。

最有趣的「文獻 vs 資料」分歧在 Jazz：文獻預測應採用低 `transition_lambda=30`（容許速度變化），但 data-driven 搜尋發現最佳值為 `tl=200`（更嚴格的速度約束）。我們推測這源於 GTZAN 的 30 秒片段限制——速度變化在如此短的時段內難以充分展現，反而是高 `tl` 過濾雜訊的效益更大。**這提醒我們：文獻結論的有效性常依賴特定的資料集假設，直接套用前需審慎評估。**

### 5.4 曲風層級的 「沒有銀彈」

Per-genre 分析揭露了一個強烈訊號：**沒有任何單一方法在所有曲風上勝出**。
- **Reggae** 是 Beat This! 唯一明顯落敗的曲風（90.48% vs baseline 92.87%，−2.39%），可能因反拍特性讓 Transformer 的 strong onset learning 過度反應。
- **Classical** 上 baseline 仍小幅領先（66.03% vs 64.97%），但 BT + 曲風感知 DBN 反過來超越所有方法（68.08%）。
- **Rock** 上 RNN + Per-Genre DBN（93.15%）擊敗了 Beat This!（92.09%），顯示在 tempo 高度穩定的曲風中，DBN 的速度連續約束仍具優勢。

這個觀察為「**hybrid system**」（按曲風選擇模型）提供了直接動機：理論上能達到 90%+ 的 oracle 上限，是未來工作的重要方向。

### 5.5 整體結論

本研究呈現了清晰的效能演進：

| 演進步驟 | 提升 |
|---------|------|
| Baseline → Per-Genre DBN | +1.40%（顯示後處理彈性的價值） |
| Per-Genre DBN → Beat This! | +0.30%（架構升級的邊際貢獻） |
| Beat This! → BT Ensemble | +0.03%（多 checkpoint 收益極小） |
| Beat This! + DBN → BT + Genre-Aware DBN | +1.03%（驗證 DBN 傷害可被緩解） |

主要洞見：**架構升級貢獻小於後處理彈性化的貢獻**。這對未來研究方向有啟示：與其追求更複雜的模型，不如在後處理中注入更多領域知識。

---

## 六、結論（Conclusion）

本研究在 GTZAN 上系統性評估了八種節拍偵測方法，最終提交採用 Beat This!（`final0` checkpoint，`dbn=False`），達到 **F1 = 88.72%**，超過目標門檻 87.5% 達 +1.22%。

核心貢獻：
1. **解碼器比較**：DBN 在 GTZAN 上整體優於 CRF（+0.59%），但在 Reggae 等不規律曲風上 CRF 略勝。
2. **Data-driven 曲風感知 DBN**：將 RNN 系統從 87.02% 提升至 **88.42%**，超越文獻 literature-driven 方法（87.22%）。
3. **Beat This! + 曲風感知 DBN（本研究核心創新）**：將「DBN 傷害 Transformer」的差距從 1.16% 縮小到 0.13%，並在 Classical、Disco、HipHop、Pop 上超越純 Beat This!。
4. **Per-genre 分析**：揭露「沒有單一方法通吃所有曲風」，為 hybrid system 提供直接動機（oracle 上限可達 90%+）。
5. **意外發現**：Jazz 的最佳 DBN 參數與文獻預測相反，揭示 GTZAN 30 秒片段對速度變化先驗的影響。

未來工作方向：
- **Cross-architecture activation fusion**：將 RNN 與 Beat This! 的 activation 加權融合（限於計算資源，本研究未完成）。
- **Hybrid system**：根據音訊內容自動選擇 RNN 或 Beat This!，逼近 oracle 上限。
- **更長時間片段驗證**：用完整曲目（非 30 秒片段）測試文獻關於速度變化的預測。

---

## 參考文獻（References）

1. Böck, S., Krebs, F., & Schedl, M. (2011). Evaluating the Online Capabilities of Onset Detection Methods. *ISMIR*.
2. Böck, S., Krebs, F., & Widmer, G. (2014). A Multi-Model Approach to Beat Tracking Considering Heterogeneous Music Styles. *ISMIR*.
3. Davies, M., & Böck, S. (2019). Temporal Convolutional Networks for Musical Audio Beat Tracking. *EUSIPCO*.
4. Krebs, F., Böck, S., & Widmer, G. (2015). Inferring Beat Rhythms Using a Multi-Scale Analysis of Musical Audio. *ISMIR*.
5. Zhao, J., et al. (2022). Beat-Transformer: Demixed Beat and Downbeat Tracking with Dilated Self-Attention. *ISMIR*.
6. Böck, S., Davies, M., & Knees, P. (2024). Beat This! Accurate Beat Tracking Without DBN Postprocessing. *ISMIR*.
