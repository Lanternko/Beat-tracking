# FINDINGS — KR 高端 soloQ 對線分析：統一論點（2026-06-15）

> 全專案綜述 / capstone。把九節分散證據收攏成一句話 + 一張表。
> 操作入口見 [`README.md`](README.md)；各線細節見下表 doc 連結。
> 資料：9648 場 KR Challenger+GM ranked solo（queue 420）、13403 玩家、172 英雄、96480 對線列。全部 walk-forward / 控制 confound / 附 CI。

---

## 一句話

**在 KR 最頂端，matchmaking 把所有「看得見的」賽前訊號都壓到 ~50% 勝率——選角、對位、meta、甚至對線結果本身。唯一漏掉、因而殘存的賽前訊號，是 matchmaking 看不到的那一件事：這個玩家是不是在打他的招牌英雄（+11pp）。不是英雄的事，是玩家×英雄匹配的事。**

## 統一論點：什麼預測勝負、什麼不預測（全部高端 soloQ）

| 賽前訊號 | → 勝負 | 結論 | 出處 |
|---|---|---|---|
| 純 draft（選角）| AUC **≈0.50**（KR 52.5%）| 選角不決定勝負 | [draft_vs_execution.py](draft_vs_execution.py) · [RESULTS.md](repro/RESULTS.md) |
| 英雄對位 → **對線 gd@10** | R² 0.06 / AUC 0.62（**有**訊號）| draft **塑形對線** | [MATCHUP_GD10.md](repro/MATCHUP_GD10.md) §1 |
| 但「對位可預測的那塊 gd@10」→ 勝負 | AUC **0.498** | 塑形的那塊**換不到勝** | MATCHUP_GD10 §4 |
| off-meta 選角 → 勝負 | **≈0**（只傷對線、不傷勝率）| meta 不是勝負軸 | [offmeta_causal.py](offmeta_causal.py)（finding #5）|
| meta / 選角偏好 | 寬池子、pick⊥勝負 | 「大家挑最強」不成立 | [predict_picks.py](predict_picks.py)（#4）|
| 玩家**對線實力** → 勝負 | AUC **0.516** [.506,.525]（弱）| matchmaking 壓到殘留一絲 | MATCHUP_GD10 §4（§7）|
| **玩家×英雄招牌（main 不 main）** → 勝負 | **+11.1pp** [9.0,13.1]（控玩家+英雄）| **唯一大槓桿** | [player_identity.py](player_identity.py)（menu B、#6）|

→ 一路向右讀：能預測勝負的東西越來越少、越來越「玩家身份」。**選角/對位/meta 全 ⊥ 勝負；只有『這人有沒有 main 這隻』穿得過 matchmaking。**

## 為什麼（機制）

兩個地基洞見（survey，詳見 auto-memory `sr-prior-art-survey`）：

1. **matchmaking 天花板**：遊戲按 MMR 把每場做成 ~50/50，**賽前訊號被刻意抹平**，~52–55% 是資訊論上限、非模型爛。賽前 predictor（draft/MMR）被抹平 → 上限低；賽中 predictor（gd@10、物件）是 execution 下游、沒被抹平 → 看似準（其實在量已跑出的結果）。**殘存賽前訊號只活在 matchmaking 量不到的維度**——而 rank/MMR 它都平衡了，唯獨「這人這場打的是不是他的招牌」它看不到。
2. **指標哲學**：acc 不是重點，log-loss + Brier 分解（Resolution）+ 校準（ECE，標竿 LoLDraftAI 0.0088）才是；每勝率附 Beta CI（高端帶窄→畫區間帶別假裝排名）。

對標背書：ProjektZero 職業 player-Elo **64%** vs 高端 draft **52%**（[RESULTS.md](repro/RESULTS.md)）——**player 強度 ≫ draft**，與本專案三軸結論同向。

## menu A（對線 / gd@10 預測）——本線已徹底封棺

完整報告 [`repro/MATCHUP_GD10.md`](repro/MATCHUP_GD10.md)（腳本 [matchup_gd10.py](matchup_gd10.py) §6 · [gd10_ceiling.py](gd10_ceiling.py) §7 · [gd10_teamctx.py](gd10_teamctx.py) §8 · [gd10_gbm.py](gd10_gbm.py) §9）。

問題：draft 既然 ⊥ 勝負，那 draft 決不決定「對線結果(gd@10)」本身？答案 = 決定一點點、且那一點點換不到勝：

| 槓桿 | 對 gd@10 oos R² | 結論 |
|---|---|---|
| 英雄對位（加性） | **0.058** / AUC 0.62 | 唯一非零來源（CS 差 cd10 更高 0.152）|
| + 玩家對線實力 | +0.003 | ≈0（高端窄帶壓縮）|
| + 打野壓力 / 隊伍 draft | +0.000 | 零跨路外溢、對線是孤島 |
| + non-linear（GBM 掃 4 組正則化）| **負**（−.03~−.04）| 封棺：champ-pair 交互＝純噪音 |

**高端 gd@10 預測天花板 ≈6% R² / 0.62 AUC，純對位已抵達；94% 是賽前碰不到的臨場執行。** counter（專剋）殘差 split-half r=.072≈雜訊——對線 ≈ 各英雄單體強度相加。閉合迴路：actual gd@10→win 0.639（含臨場）但 predicted(純 draft)→win **0.498**＝**選角塑形對線、塑形那塊換不到勝**，唯一微弱流向勝負的是玩家對線實力(0.516)。

## 還開放的前沿

- **menu B 主線**（進行中，另一 session）：player latent skill（TrueSkill on PUUID、⚠️time-split 防 leakage）、拆 treatment(練) vs selection(選你擅長的)、招牌池廣度 vs 勝率/variance。詳見 [`HANDOFF-menu-b-mastery-2026-06-15.md`](HANDOFF-menu-b-mastery-2026-06-15.md)。
- **menu D 賽中動態**：live win-probability、翻盤金錢門檻、tempo/objective——概念缺口最大（ProjektZero 缺 counter-pick/momentum）。
- **menu E 產品**：對線 tier list（matchup_gd10 的 θ 直接就是）+ comp synergy + 區間帶站。
- **menu C 職業賽**：已大致完成（[`repro/DRAFT_META.md`](repro/DRAFT_META.md)：pro 選角 top-1 28.5%、可用性約束主宰、身份訊號弱、meta 半衰期 8 patch）。

## 細節 doc 地圖

| 線 | doc | 一句話 |
|---|---|---|
| 綜述（本檔）| `FINDINGS.md` | 統一論點 + 證據表 |
| 複現 / 前向 / 校準 | [repro/RESULTS.md](repro/RESULTS.md) · [repro/HANDOFF-2026-06-15.md](repro/HANDOFF-2026-06-15.md) | ProjektZero 64% vs draft 52%、matchmaking 天花板 |
| menu A 對線 gd@10 | [repro/MATCHUP_GD10.md](repro/MATCHUP_GD10.md) | draft 塑形對線、塑形那塊換不到勝；天花板封棺 |
| menu B 玩家身份 | [HANDOFF-menu-b-mastery-2026-06-15.md](HANDOFF-menu-b-mastery-2026-06-15.md) · [HANDOFF-offmeta-causal-2026-06-15.md](HANDOFF-offmeta-causal-2026-06-15.md) | 招牌 +11pp、familiarity≫meta |
| 選角預測 / meta | [HANDOFF-pick-prediction-2026-06-15.md](HANDOFF-pick-prediction-2026-06-15.md) | soloQ top-1 15%、有效英雄池 |
| menu C 職業 draft / meta | [repro/DRAFT_META.md](repro/DRAFT_META.md) | pro top-1 28.5%、可用性主宰 |
