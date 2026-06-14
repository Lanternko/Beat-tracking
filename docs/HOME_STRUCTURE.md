# Home Directory — Claude Code 上手指南

## 使用者概覽

**身份**：音訊/音樂 AI 研究者（Lanternko）
**主要研究**：文字驅動音訊生成（MeanAudio）、音樂語義保真度（DAC codec）、音樂字幕生成（LP-MusicCaps）
**硬體**：RTX 5090（33.67 GB VRAM）、外接 HDD `/mnt/HDD/kojiek/`
**語言偏好**：繁體中文

---

## 目錄總覽

### 核心專案

| 目錄 | 用途 | 狀態 |
|------|------|------|
| `MeanAudio/` | 文字→音訊生成（主專案，Phase 6 V2） | **活躍開發中** |
| `lp-music-caps/` | LLM 音樂字幕生成（ISMIR 2023 論文實作） | 工具/依賴 |

### 實驗室研究（`research/`）

每個實驗自成一體，各自有 `core/`（驗證過的腳本）和 `output/`（結果）。
日常新腳本直接寫在實驗根目錄，驗證後再 `mv` 進 `core/`。

| 目錄 | 用途 | 時期 |
|------|------|------|
| `research/dac_codec/` | DAC codec 語義保真度研究 | 最早 |
| `research/music_cleaning/` | 音訊品質評估與資料清洗 | 中期 |
| `research/meanaudio_eval/` | MeanAudio 評估（RTF、偏差分析） | 最新 |
| `research/meanaudio_training/` | 訓練資料準備（NPZ、TSV、splits） | 持續 |

### 個人專案（`side_projects/`）

實驗室以外的專案（課堂作業、個人興趣）。
Git repo: `https://github.com/Lanternko/side_projects` (private)

| 目錄 | 用途 |
|------|------|
| `side_projects/apps/discord-social-preview-bot/` | Discord social preview bot |
| `side_projects/music-ai/karaoke-jp/` | 日文 karaoke timing / rendering pipeline |
| `side_projects/music-ai/lyrics-transcription-benchmark/` | 日文歌詞轉錄 benchmark |
| `side_projects/coursework/audio/GTZAN/` | 神經網路課 — 音樂類型分類 |
| `side_projects/coursework/audio/beat-tracking/` | Beat tracking 作業 |
| `side_projects/coursework/nlp/Qwen3-SLU-for-NLP/` | Qwen3-SLU 作業 |
| `side_projects/coursework/nlp/nlp_csc_assignment_11502_2/` | 中文拼字檢查作業 |
| `side_projects/reference-repos/chinese-spelling-check/` | CSC 作業/實驗參考 repo |
| `side_projects/_archive/` | 舊備份與作業成果封存 |

### 資料入口（`data/`）

所有外部資料 symlink 集中在此。

| Symlink | 指向 |
|---------|------|
| `data/meta_all.json` | `/home/hsiehyian/projectB/data/Jamendo/meta_all.json` |
| `data/segments_no_vocals` | `/home/hsiehyian/dataset/segments_no_vocals` |
| `data/music_semantic_fidelity` | `/mnt/HDD/kojiek/music_semantic_fidelity` |
| `MeanAudio/exps/` | `/mnt/HDD/kojiek/meanaudio_exps` |

### 其他

| 目錄 | 用途 |
|------|------|
| `logs/` | 訓練日誌（Phase 3~6，按 phase 命名） |
| `archives/` | 歷史實驗、舊腳本、備份 |
| `venvs/dac/` | **主要 Python 環境** |
| `venvs/gtzan/` | GTZAN 分類用 |
| `venvs/whisper/` | Whisper 轉錄用 |
| `bin/` | GitHub CLI (`gh`) |
| `scripts` → | `/mnt/HDD/kojiek/scripts`（共用腳本庫） |

---

## 常用工作流程

### 啟動環境

```bash
source ~/venvs/dac/bin/activate
export CUDA_VISIBLE_DEVICES=0
```

### MeanAudio 訓練（主要工作）

```bash
tmux new -s phase6v2
cd ~/MeanAudio && source ~/venvs/dac/bin/activate
bash train_pipeline.sh   # Stage 1 → migrate → Stage 2 → eval
```

詳見 `MeanAudio/CLAUDE.md` — 該檔案有完整的架構說明、訓練參數、eval 指令。

### 實驗室研究（日常流程）

```bash
cd ~/research/dac_codec/       # 進入實驗目錄
vim new_script.py              # 新腳本直接寫在根目錄
python new_script.py           # 跑完驗證
mv new_script.py core/         # 確認可用後歸入 core/
```

### 查看訓練進度

```bash
tail -f ~/logs/phase6_v2_stage2_200000.log
```

---

## 專案間關係

```
LP-MusicCaps（字幕模型）
    ↓ 提供語義評估能力
research/dac_codec/（DAC 語義保真度）
research/music_cleaning/（資料品質）
research/meanaudio_eval/（模型評估）
    ↓ 研究成果回饋至
MeanAudio（音訊生成）← 主專案
    ↑ 資料來自 data/（Jamendo symlinks）
```

---

## Git

| Repo | URL | 類型 |
|------|-----|------|
| MeanAudio | `https://github.com/Lanternko/MeanAudio.git` | 主專案 |
| side_projects | `https://github.com/Lanternko/side_projects.git` | 個人專案（private） |

- Git identity: `lanternko <jerry86012@gmail.com>`
- Auth: token 已存在 credential store

---

## 注意事項

- `~/venvs/dac/` 是最常用的環境，幾乎所有音訊相關工作都用它
- 大型資料和 checkpoint 都在外接 HDD（`/mnt/HDD/kojiek/`），不在 home 目錄
- 訓練日誌按 phase 命名存在 `~/logs/`
- `archives/music_cleaning_backup/` 有 tar.gz 備份，重複資料夾已清除
- `archives/legacy_scripts/` 存放早期可行性測試腳本（已整合進 MeanAudio）
