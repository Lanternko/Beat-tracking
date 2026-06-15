"""Stage 2：從 KR Challenger + GrandMaster 收集對局 -> 算對線指標 -> 寫進 SQLite。

用法：python3 build_dataset.py [目標場數=1500]
- 多 key 自動 ~2× 速度（見 riot_client）。
- 可續傳：原始回應走磁碟快取，DB 用 INSERT OR REPLACE 冪等。
- 不在 crawl 階段過濾 patch（全收、存 patch 欄位），交給 analyze 篩，避免浪費 detail call。
"""
import sys

import config
import db
from riot_client import RiotClient

TARGET_GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
IDS_PER_PLAYER = 100  # 每位玩家抓多少近期 match id（大 crawl 要拉高才湊得到唯一場次）
MIN_DURATION = 840  # 秒；<14min 視為 remake，跳過（確保 @10/@14 存在）
POSITIONS = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
LEAGUE_ENDPOINTS = (
    f"/lol/league/v4/challengerleagues/by-queue/{config.RANKED_SOLO_STR}",
    f"/lol/league/v4/grandmasterleagues/by-queue/{config.RANKED_SOLO_STR}",
)


def patch_of(gv):
    return ".".join(gv.split(".")[:2])  # "16.12.785.1316" -> "16.12"


def frame_diff(frames, pid_self, pid_opp, minute):
    """(golddiff, xpdiff, csdiff) @minute，自己視角。"""
    idx = minute if len(frames) > minute else len(frames) - 1
    pf = frames[idx]["participantFrames"]
    s, o = pf[str(pid_self)], pf[str(pid_opp)]
    cs_s = s.get("minionsKilled", 0) + s.get("jungleMinionsKilled", 0)
    cs_o = o.get("minionsKilled", 0) + o.get("jungleMinionsKilled", 0)
    return (s["totalGold"] - o["totalGold"], s["xp"] - o["xp"], cs_s - cs_o)


def collect_match_ids(c, want):
    players = []
    for ep in LEAGUE_ENDPOINTS:
        league = c.get(config.PLATFORM_HOST, ep)
        players += sorted(league["entries"], key=lambda e: -e.get("leaguePoints", 0))
    print(f"seed pool: {len(players)} players (Challenger + GrandMaster)", flush=True)
    seen, order = set(), []
    for e in players:
        ids = c.get(
            config.REGIONAL_HOST,
            f"/lol/match/v5/matches/by-puuid/{e['puuid']}/ids",
            {"queue": config.QUEUE_RANKED_SOLO, "type": "ranked", "start": 0, "count": IDS_PER_PLAYER},
        )
        for m in ids:
            if m not in seen:
                seen.add(m)
                order.append(m)
        if len(order) >= want:
            break
    return order


def main():
    all_keys = config.load_api_keys()
    c_ids = RiotClient(api_keys=[all_keys[0]])  # 收 ids：puuid 綁 key，只能用單一把
    c_match = RiotClient()                       # 抓 match：matchId 不綁 key，多 key 輪替 ~Nx
    print(f"ids: 1 key | match fetch: {len(c_match.keys)} keys (~{len(c_match.keys)}x)", flush=True)
    conn = db.connect()
    db.init(conn)

    mids = collect_match_ids(c_ids, int(TARGET_GAMES * 1.4))
    print(f"collected {len(mids)} unique match ids; target {TARGET_GAMES} games", flush=True)

    kept = 0
    for mid in mids:
        if kept >= TARGET_GAMES:
            break
        try:
            detail = c_match.get(config.REGIONAL_HOST, f"/lol/match/v5/matches/{mid}")
            info = detail["info"]
            if info.get("queueId") != config.QUEUE_RANKED_SOLO:
                continue
            if info.get("gameDuration", 0) < MIN_DURATION:
                continue
            timeline = c_match.get(config.REGIONAL_HOST, f"/lol/match/v5/matches/{mid}/timeline")
        except Exception as ex:
            print(f"  skip {mid}: {ex}", flush=True)
            continue

        parts = info["participants"]
        frames = timeline["info"]["frames"]
        pos_map = {(p["teamId"], p.get("teamPosition")): p for p in parts}
        win_side = 100 if [t for t in info["teams"] if t["teamId"] == 100][0]["win"] else 200

        db.insert_match(conn, mid, patch_of(info["gameVersion"]), info["queueId"],
                        info["gameDuration"], win_side, info.get("gameCreation"))
        for p in parts:
            db.insert_participant(conn, mid, p)
            pos = p.get("teamPosition")
            opp = pos_map.get((200 if p["teamId"] == 100 else 100, pos))
            if opp and pos in POSITIONS:
                diffs = (*frame_diff(frames, p["participantId"], opp["participantId"], 10),
                         *frame_diff(frames, p["participantId"], opp["participantId"], 14))
                db.insert_laning(conn, mid, p, opp, diffs)
        conn.commit()
        kept += 1
        if kept % 50 == 0:
            print(f"  parsed {kept}/{TARGET_GAMES}  "
                  f"(api_calls={c_ids.n_calls + c_match.n_calls}, "
                  f"cache={c_ids.n_cache_hits + c_match.n_cache_hits})", flush=True)

    print(f"DONE: {kept} games -> {config.DB_PATH}  "
          f"(api_calls={c_ids.n_calls + c_match.n_calls}, "
          f"cache_hits={c_ids.n_cache_hits + c_match.n_cache_hits})", flush=True)


if __name__ == "__main__":
    main()
