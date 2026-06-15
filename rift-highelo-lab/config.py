"""KR 高端 soloQ 設定：routing、queue、rate limit、API key 載入。"""
import os
from pathlib import Path

# Routing：KR 的 platform host 與 regional host 不同
PLATFORM_HOST = "kr.api.riotgames.com"     # league-v4 / summoner-v4 (platform routing)
REGIONAL_HOST = "asia.api.riotgames.com"   # match-v5 / account-v5  (KR -> asia regional)

QUEUE_RANKED_SOLO = 420
RANKED_SOLO_STR = "RANKED_SOLO_5x5"

# dev key 限制：20 req/s 且 100 req/2min（後者是瓶頸）
RATE_LIMITS = [(20, 1), (100, 120)]

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
DB_PATH = DATA_DIR / "lol.db"


KEY_VARS = ("RIOT_API_KEY", "RIOT_API_KEY2", "RIOT_API_KEY3", "RIOT_API_KEY4")


def load_api_keys() -> list:
    """收集所有可用的 key（環境變數優先，再補 .env），保序去重。多把 = 多份額度。"""
    keys = []
    for var in KEY_VARS:
        v = os.environ.get(var)
        if v:
            keys.append(v.strip())
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            for var in KEY_VARS:
                if line.startswith(var + "="):
                    keys.append(line.split("=", 1)[1].strip())
    seen, out = set(), []
    for k in keys:
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    if not out:
        raise RuntimeError("找不到任何 RIOT_API_KEY；設環境變數或寫進 .env")
    return out


def load_api_key() -> str:
    return load_api_keys()[0]
