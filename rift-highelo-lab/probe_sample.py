"""Gold-first 第一抓：1 位 KR Challenger -> 1 場 ranked solo -> detail + timeline，
落地原始 JSON 並驗證 top lane 的 golddiff@10（確認結構與口徑後才寫正式 parser）。
"""
import json
from pathlib import Path

import config
from riot_client import RiotClient


def main():
    c = RiotClient()
    raw = Path(config.RAW_DIR)
    raw.mkdir(parents=True, exist_ok=True)

    # 1) Challenger league (KR)
    league = c.get(
        config.PLATFORM_HOST,
        f"/lol/league/v4/challengerleagues/by-queue/{config.RANKED_SOLO_STR}",
    )
    entries = league.get("entries", [])
    print(f"[1] Challenger entries: {len(entries)}")
    print(f"    first entry keys: {sorted(entries[0].keys())}")

    # 取一個 puuid（新版 league-v4 entry 直接帶 puuid；沒有就用 summoner-v4 換）
    e0 = max(entries, key=lambda e: e.get("leaguePoints", 0))  # 拿 LP 最高那位
    puuid = e0.get("puuid")
    if not puuid:
        s = c.get(config.PLATFORM_HOST, f"/lol/summoner/v4/summoners/{e0['summonerId']}")
        puuid = s["puuid"]
    print(f"[2] top-LP challenger: LP={e0.get('leaguePoints')} puuid={puuid[:16]}...")

    # 2) 最近的 ranked solo match ids
    ids = c.get(
        config.REGIONAL_HOST,
        f"/lol/match/v5/matches/by-puuid/{puuid}/ids",
        {"queue": config.QUEUE_RANKED_SOLO, "type": "ranked", "start": 0, "count": 20},
    )
    print(f"[3] recent ranked-solo matches: {len(ids)}  e.g. {ids[0] if ids else 'NONE'}")
    if not ids:
        print("這位玩家近期沒有 solo 排位；換一位再試。")
        return
    mid = ids[0]

    # 3) match detail + timeline
    detail = c.get(config.REGIONAL_HOST, f"/lol/match/v5/matches/{mid}")
    timeline = c.get(config.REGIONAL_HOST, f"/lol/match/v5/matches/{mid}/timeline")
    (raw / f"match_{mid}.json").write_text(json.dumps(detail))
    (raw / f"timeline_{mid}.json").write_text(json.dumps(timeline))

    info = detail["info"]
    parts = info["participants"]
    by_pid = {p["participantId"]: p for p in parts}
    print(f"[4] match {mid}")
    print(
        f"    gameVersion={info['gameVersion']}  queueId={info['queueId']}  "
        f"duration={info['gameDuration']}s"
    )
    print("    lineup (pid / team / pos / champ):")
    for p in parts:
        print(
            f"      pid{p['participantId']:<2} team{p['teamId']:<3} "
            f"{p.get('teamPosition', '?'):<8} {p['championName']}"
        )

    # 4) timeline 結構
    frames = timeline["info"]["frames"]
    interval = timeline["info"].get("frameInterval", 60000)
    print(f"[5] timeline: {len(frames)} frames, interval={interval}ms")
    pf_sample = frames[min(10, len(frames) - 1)]["participantFrames"]["1"]
    print(f"    participantFrame keys: {sorted(pf_sample.keys())}")

    # 5) 驗證：top lane golddiff/xpdiff/csdiff @10
    f10 = 10 if len(frames) > 10 else len(frames) - 1
    pf10 = frames[f10]["participantFrames"]
    tops = {p["teamId"]: p["participantId"] for p in parts if p.get("teamPosition") == "TOP"}
    if len(tops) == 2:
        b, r = tops[100], tops[200]

        def cs(pid):
            f = pf10[str(pid)]
            return f.get("minionsKilled", 0) + f.get("jungleMinionsKilled", 0)

        gd = pf10[str(b)]["totalGold"] - pf10[str(r)]["totalGold"]
        xd = pf10[str(b)]["xp"] - pf10[str(r)]["xp"]
        cd = cs(b) - cs(r)
        print(f"[6] TOP @~{f10}min  {by_pid[b]['championName']}(藍) vs {by_pid[r]['championName']}(紅)")
        print(f"    golddiff={gd:+d}  xpdiff={xd:+d}  csdiff={cd:+d}  (藍視角)")
    else:
        print("[6] 這場沒有乾淨的雙方 TOP；換場再驗。")

    print(f"\n[stats] api_calls={c.n_calls} cache_hits={c.n_cache_hits}")
    print(f"[raw] saved to {raw}/match_{mid}.json , timeline_{mid}.json")


if __name__ == "__main__":
    main()
