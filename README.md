# side_projects

整理日期：2026-05-31

這個資料夾現在用「用途」分類，根目錄只放索引、設定、文件與大分類資料夾。

## 目錄

| 路徑 | 用途 |
| --- | --- |
| `apps/` | 可運行的個人 app / bot |
| `music-ai/` | 音樂、歌詞、karaoke 相關個人專案 |
| `coursework/` | 課堂作業與可交付成果 |
| `reference-repos/` | 作業或實驗用的外部論文/模型復現 repo |
| `_archive/` | 舊備份、交付壓縮檔、暫時不想放在工作區第一層的東西 |
| `docs/` | 工作區說明與背景文件 |

## 專案索引

| 路徑 | 內容 |
| --- | --- |
| `apps/discord-social-preview-bot/` | Discord social preview bot |
| `music-ai/karaoke-jp/` | 日文 karaoke timing / rendering pipeline |
| `music-ai/lyrics-transcription-benchmark/` | 日文歌詞轉錄 benchmark |
| `coursework/audio/GTZAN/` | 神經網路課：GTZAN 音樂類型分類 |
| `coursework/audio/beat-tracking/` | Beat tracking 作業與 submission |
| `coursework/nlp/Qwen3-SLU-for-NLP/` | HW04 Qwen3-SLU baseline/reproduction |
| `coursework/nlp/nlp_csc_assignment_11502_2/` | 中文拼字檢查作業 |
| `reference-repos/chinese-spelling-check/DR-CSC/` | Chinese spelling check 參考 repo |
| `reference-repos/chinese-spelling-check/SCOPE/` | Chinese spelling check 參考 repo |
| `_archive/backups/discord-social-preview-bot.bak/` | Discord bot 舊備份 |
| `_archive/submissions/HW04_Qwen3_SLU_results.tar.gz` | Qwen3-SLU 作業結果壓縮檔 |

## 清理原則

- 可重建的 dependency/cache 直接移除，例如 `node_modules/`、`.venv/`、`__pycache__/`、`.pytest_cache/`、`.snakemake/`、`.DS_Store`、舊 `*.log` 與 `*.pid`。
- 專案本體、作業結果、模型輸出先保留；大型輸出如果確定不用，再針對性刪。
- 外部 repo 和備份不要混在根目錄第一層，避免日常找專案時被淹沒。
