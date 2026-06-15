"""SQLite schema：對線研究用的三張表。crawl 一次（慢），analyze 多次（快）。"""
import sqlite3

import config


def connect():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    return conn


def init(conn):
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS matches(
          match_id TEXT PRIMARY KEY, patch TEXT, queue INT, duration INT,
          win_side INT, game_creation INT);

        CREATE TABLE IF NOT EXISTS participants(
          match_id TEXT, puuid TEXT, champion TEXT, team_id INT, team_position TEXT,
          win INT, kills INT, deaths INT, assists INT,
          PRIMARY KEY(match_id, puuid));

        -- 對線軸：對位 = 同 team_position 的敵方；diff 皆為「自己 - 對手」
        CREATE TABLE IF NOT EXISTS laning(
          match_id TEXT, puuid TEXT, champion TEXT, team_position TEXT, win INT,
          opp_puuid TEXT, opp_champion TEXT,
          gd10 INT, xd10 INT, cd10 INT, gd14 INT, xd14 INT, cd14 INT,
          PRIMARY KEY(match_id, puuid));
        """
    )
    conn.commit()


def insert_match(conn, mid, patch, queue, duration, win_side, gc):
    conn.execute(
        "INSERT OR REPLACE INTO matches VALUES(?,?,?,?,?,?)",
        (mid, patch, queue, duration, win_side, gc),
    )


def insert_participant(conn, mid, p):
    conn.execute(
        "INSERT OR REPLACE INTO participants VALUES(?,?,?,?,?,?,?,?,?)",
        (mid, p["puuid"], p["championName"], p["teamId"], p.get("teamPosition"),
         int(p["win"]), p["kills"], p["deaths"], p["assists"]),
    )


def insert_laning(conn, mid, p, opp, diffs):
    gd10, xd10, cd10, gd14, xd14, cd14 = diffs
    conn.execute(
        "INSERT OR REPLACE INTO laning VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (mid, p["puuid"], p["championName"], p.get("teamPosition"), int(p["win"]),
         opp["puuid"], opp["championName"], gd10, xd10, cd10, gd14, xd14, cd14),
    )
