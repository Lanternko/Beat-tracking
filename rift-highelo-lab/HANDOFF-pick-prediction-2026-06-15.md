# 交接：§4 選角預測 / meta 收斂（2026-06-15）

新對話接這份就能續 §4 → §5。配 auto-memory `sr_pick_prediction`、`sr_highelo_project`、`sr_prior_art_survey`、`sr_repro_results`。
腳本 = [`predict_picks.py`](predict_picks.py)（根目錄，9648 場 `data/lol.db`，**16s 可重跑、確定性 SEED=42**）。

---

## 這次做了什麼（一句話）

建 goal 的樞紐 §4：用 **masked-completion** 量「高端 picks 多可預測」= 選角預測率 + meta 收斂度（有效英雄池），
並把「off-meta」操作化（模型給實際選角的機率），為 §5（偏離 meta 對勝負的影響）鋪路。

## 關鍵數字（at a glance）

| 主題 | 結果 |
|---|---|
| 選角預測率（test, 情境 LR）| top-1 **15.0%** [14.6,15.4] / top-3 30.6% / top-5 41.2%（隨機 0.58%）|
| 邊際 baseline（純人氣）| top-1 11.8% / top-5 37.5% → **context 只 +3pp** |
| per role top-1（情境）| BOT 22.8 / SUP 20.0 / JNG 15.6 / TOP 9.6 / **MID 7.1**（最不可測）|
| meta 有效英雄池（2^H）| **BOT 25 < JNG 31 ≈ SUP 31 < MID 48 < TOP 51** |
| 跨 patch（16.8–16.12）| 有效池穩定、無劇烈收斂趨勢 |
| off-meta（§5 預告，未控玩家）| 模型機率十分位 vs 勝率 **全平 ~50%** = 無 off-meta 勝率稅 |

**三句結論**：① 高端選角由 tier-list 人氣主導、不繞 lobby 選 counter（context 僅 +3pp）。
② meta 收斂是**分路現象**（bot/野/輔已解 ~25-31 隻，top/mid 大開 ~50）。
③ 選得多 meta / 多偏門跟贏不贏沒關係（勝率全平 ~50%）= draft⊥勝率的「選角可預測度」版再現。

## 任務定義 / 口徑（為什麼這樣做）

- **masked-completion**：遮 10 picks 之一，從其餘 9（champ + role + 敵我）+ 目標 role + patch 預測被遮英雄。
- **為何不是逐手預測**：Riot soloQ 資料**沒有 pick 順序**（LoLDraftAI 也因此隨機遮罩）。逐手預測只能在職業賽（OE 有 `pick1`–`pick5`/`firstPick`）做。
- **切 train/test 依 match**（同場 9 隻共享 → 避免洩漏），80/20，SEED=42。
- **兩模型**：邊際 `P(champ|role,patch)`（人氣 baseline）vs **multinomial LR**（情境，特徵 = 其餘 9 的 `champ×role×敵我` one-hot + 目標 role + patch；saga, C=3.0）。
- **meta 收斂** = 每 `(role,patch)` pick 分布的 perplexity `2^H` = 「有效英雄池大小」（model-free）。

## 資料限制（誠實，接手必讀）

- **無 pick 順序** → masked-completion 非 sequential（Riot 限制）。
- **無 ban**（`db.py` 沒存）→ 但 **7.9G raw cache（`data/cache`）有 ban**，可再抽 `ban1`–`ban5` 加進來（被 ban 的英雄不能選 = 該移出候選 + 當特徵）。
- **單一 elo 帶**（全 Challenger+GM，DB 沒存 per-player tier）→ **無法測「低 elo 更可預測」**。要驗 elo 梯度得爬 Emerald/Diamond 或用公開多-elo 資料。

## 怎麼重跑

```bash
cd ~/side_projects/rift-highelo-lab
python3 predict_picks.py     # 16s，不碰 API，確定性（SEED=42）
```

環境：python3 + numpy/scipy/sklearn（`~/venvs/dac` 已裝）。

## 下一步（推薦序）

1. **§5 正式版（goal 核心、最該做）**：控玩家強度後看 off-meta 的**因果**。
   - off-meta 定義：用 `predict_picks` 的 `p_true`（已算，低 = 偏離）或該 (role,patch) pick rate 低。
   - **控 player**：PUUID fixed-effect / 只取重複出現玩家 / 用該玩家整體勝率當 covariate。資料：`participants.puuid`（13403 人、seed 玩家場次多）、`laning.gd10`。
   - 量 3 個 outcome：① 對線 `gd@10`（off-meta 傷不傷線）② 勝率（控玩家後還剩多少）③ variance。
   - 預期：**傷線、不太傷勝率**（天花板 + 高端可恢復），但要排 smurf/one-trick confound 才算數。= menu B「玩家身份軸」首次動工。
2. **NN 版（鏡像 LoLDraftAI，看能否抬 15%）**：`embedding(champ)` + `patch embedding` + `champ×patch 交互` + MLP，masked 訓練。env `~/venvs/dac` 有 torch。判斷點：**若 NN 也只 ~15% = 預測天花板是「真選擇多樣性」不是模型弱**。
3. **跨 elo 收斂對照**：爬 Emerald/Diamond（dev key 24h 過期，先驗 key）或公開資料，測「elo↑ → picks 更可預測」（與「elo↑ → 勝負更不可預測」成漂亮對照）。
4. **加 ban**：從 raw cache 抽 ban，移出候選 + 當特徵，提升預測 + 更真實。

## LoLDraftAI 開源挖點（不必逆向）

[Looyyd/loldraftai-monorepo-public](https://github.com/Looyyd/loldraftai-monorepo-public) 的 `apps/machine-learning`：
meta 編碼 = champion embedding + **patch embedding + champion×patch 交互** + masked partial draft + duration-bucket heads（0-25/25-30/30-35/35+ 分）+ `adapt_model.py` 續訓。
**值得抄三個**：patch embedding、champ×patch 交互、masked-draft 訓練（我們已用 masked，缺前兩個）。自報 SoloQ ~56% / Pro ~57%（Emerald+）。

## 未 commit

`predict_picks.py` + README/memory 改動 + 本 handoff 都在工作樹（branch `organize/first-cleanup`），**要 commit 喊一聲**。
