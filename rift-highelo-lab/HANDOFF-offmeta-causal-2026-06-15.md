# 交接：§5 偏離 meta 對勝負的因果（控玩家強度）（2026-06-15）

新對話接這份就能續 §5 → §6。配 auto-memory `sr_pick_prediction`、`sr_highelo_project`、`sr_prior_art_survey`、`sr_repro_results`。
腳本 = [`offmeta_causal.py`](offmeta_causal.py)（根目錄，96480 picks `data/lol.db`，**0.5s 可重跑、確定性 SEED=42、laning 覆蓋 100%**）。
這是 **menu B（玩家身份軸）首次動工**，也是 §4 `predict_picks.py` 的「off-meta 預告」收尾成因果。

---

## 這次做了什麼（一句話）

把「off-meta 傷不傷勝負」從 §4 的相關（機率十分位 vs 勝率全平）推進到**控玩家強度後的因果**，
發現 **真正的軸是 familiarity（玩不玩得熟）、不是 meta**：off-meta 本身只傷對線（英雄屬性）、幾乎不傷勝率。

## 關鍵數字（at a glance）

| 主題 | 結果 |
|---|---|
| RAW（不控玩家）| off-meta 傷線（gd10 core +32 → pocket −89，單調）、幾乎不傷勝率（50.1 → 49.5%）|
| within-player FE（≥10 場、減玩家均值）| gd10 **−27 金 / 十倍冷門** CI[−44,−8]（＝RAW −61 的**一半**）；win −1.31pp CI[−2.26,−0.33]，但離散 Δwin −0.52pp CI[−2.52,+1.26]（含 0）|
| player 勝率 covariate（LOO）| 係數僅 **0.04** ＝單 elo 帶玩家自己勝率幾乎不預測下一場（matchmaking 天花板再現）|
| variance | sd(gd10) 全 ~860–874 ＝**無 boom-or-bust**，off-meta 是系統性輸線但可救 |

**meta × familiarity 2×3（限 ≥10 場玩家；win% (gd10)）— 全篇核心：**

|  | unfam <10% | mid 10–30% | fam ≥30%（one-trick）|
|---|---|---|---|
| **on-meta ≥5%** | 48.6% (+33) | 51.5% (+94) | **53.1% (+61)** |
| **off-meta <1%** | 45.7% (−21) | 52.0% (−10) | **53.1% (−17)** |

**autofill 穩健性**（autofill ≈ 該場 role ≠ 該玩家眾數 role）：不熟 off-meta × 非 autofill 仍 **46.0%（gd10 +4）**、× autofill 44.9%（gd10 −85）→ 那格勝率稅**不是** autofill 假象。參考：autofill 率 23%、勝率 48.8% vs 主位 51.0%。

**五句結論**：
① **真軸是 familiarity，不是 meta**：同 meta 層內 unfam→fam 勝率 +4.5pp(on)/+7.4pp(off)；**熟練 off-meta one-trick 53.1% ＝ 熟練 on-meta 53.1%**，off-meta 本身對勝率幾乎無主效應。
② off-meta 唯一的勝率稅是 **unfam × off-meta 交互**（−2.9pp，48.6→45.7，~3.1 SE，扛得住 autofill）＝最虧的是「第一次玩冷門英雄」。
③ off-meta **傷線是英雄屬性**：整列 gd10 負、跟 on-meta 差 ~50–100 金、每 familiarity 層都在；familiarity 不修對線（off-meta 招牌仍 −17）但把輸掉的線**轉換成勝**（53.1%）＝ **finding #1（對線⊥勝率）的玩家層級版**。非 autofill 的不熟 off-meta 甚至**對線打平(+4)卻只贏 46%**＝「不輸線但輸遊戲」，英雄知識落差顯現在線後。
④ **修正跑前的中介假設**：off-meta 不是「比較常不熟」（P(unfam) 只 42%→46%），反而平均 familiarity **更高**（0.32 vs 0.16）＝被 one-trick 專精者主導，不是被實驗者主導。所以是純交互、非 mediation。
⑤ 控玩家把 RAW 傷線砍半（FE −27 vs RAW −61）；player 勝率係數 0.04 再現 matchmaking 天花板；無 variance 膨脹。

## 任務定義 / 口徑（為什麼這樣做）

- **off-meta 度量**（model-free，每個 pick 都有 → 撐得起 within-player 控制）：該英雄在 `(role,patch)` 的 pick rate；`off_score = -log10(pick_rate)`。bin：core ≥5% / playable 1–5% / off-meta 0.3–1% / pocket <0.3%。
- **familiarity** = 該英雄佔該玩家（crawl 窗內）場次比；**限 ≥10 場玩家**才有意義（1–2 場玩家 familiarity 恆=100% 是雜訊）。
- **三層控玩家**（重點＝看 RAW 的 off-meta 懲罰，控玩家後縮多少 / 消不消失）：① RAW ② within-player FE（減玩家均值 + cluster(player)-bootstrap CI）③ player 勝率 LOO covariate。
- **autofill** ≈ 該場 `team_position` ≠ 該玩家眾數 role（DB 沒存 autofill flag，這是近似）。
- **三 outcome**：對線 `gd@10`、`win`、variance（sd）。

## 資料限制（誠實，接手必讀）

- **familiarity 只在 crawl 窗（median 7 場）算** → 低估真實熟練度（玩家歷史可能 500 場），低場玩家噪 → 已用 ≥10 場門檻緩解，但 ≥30% 招牌的真熟練仍可能被低估。
- **familiarity 主效應有 survivorship**：「你只 main 你會贏的英雄」→ +7pp 的 unfam→fam 勝率梯度**有一部分是選擇而非因果**。within-player FE 控了玩家、沒控「玩家×英雄」選擇。**我不宣稱那 +7pp 全是因果**；但 off-meta 結論（②③）不依賴它。
- **單一 elo 帶**（全 Challenger+GM）：player 勝率 covariate 係數 0.04 一部分是 matchmaking 壓縮、一部分是測量誤差（多數玩家場次少 → 勝率噪 → 衰減）。FE 法比 covariate 法穩。
- **off-meta 內 mid-familiarity 的 gd10 非單調**（−21/−10/−17）＝噪，別過度解讀。
- masked/ban 等 §4 的限制不適用（§5 是觀測對局結果，非預測 picks）。

## 怎麼重跑

```bash
cd ~/side_projects/rift-highelo-lab
python3 offmeta_causal.py     # 0.5s，不碰 API，確定性（SEED=42）
```

環境：python3 + numpy（`~/venvs/dac` 已裝；`~/venvs/dac/bin/python3 offmeta_causal.py`）。

## 下一步（推薦序）

1. **champ fixed-effect 版（讓 familiarity 主效應站穩，最該做）**：現在 FE 只控玩家、沒控英雄。加 **champ FE** 或 `champ×player` within 估計，分離「英雄屬性（off-meta 弱線）」vs「玩家×英雄選擇 survivorship」。預期：champ-FE 後 off-meta 傷線主效應被英雄吸收、familiarity 殘差才是乾淨的「熟練紅利」。
2. **真 menu B：player latent skill**：對 PUUID 跑 TrueSkill/Elo（對標 ProjektZero player-Elo），取「控玩家強度後的殘差訊號」——smurf 偵測（新帳號高勝率殘差）、per-player stability/variance、招牌池廣度 vs 勝率。⚠️ 記 `sr_prior_art_survey` 的雷：玩家歷史特徵 AUC 0.97 是 leakage 教訓，務必 time-split。
3. **產品洞見**：把「高端＝熟練度 ≫ 選角」做成一句話 + 每路「最被低估的 off-meta 招牌池」（high familiarity、low pick rate、≥50% 勝率的英雄）。
4. **補 familiarity 窗限**：抓 champion-mastery API 或玩家歷史，把 familiarity 從「crawl 窗內」升級成「真實累積熟練」，重測梯度。

## 與既有發現的接點

- §5 的「off-meta 傷線不傷勝率」＝ **finding #1（對線⊥勝率）** 的玩家層級因果版。
- player 勝率係數 0.04 ＝ **`sr_repro_results`/survey 的 matchmaking 天花板**第三次現身（賽前訊號被抹平）。
- familiarity≫meta 指向 **menu B（玩家身份）才是高端殘存訊號所在**，draft（menu A）已被抹平 —— 與 survey 的地基洞見一致。
