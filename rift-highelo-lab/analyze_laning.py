"""讀 SQLite -> 對線強度榜（快、可重複跑，不碰 API）。

用法：python3 analyze_laning.py [patch=最多場的那個] [min_games=15]
"""
import sqlite3
import sys

import config


def table(conn, pos, patch, min_games):
    rows = conn.execute(
        """
        SELECT l.champion, COUNT(*) n,
               AVG(l.gd10) gd10, AVG(l.xd10) xd10, AVG(l.cd10) cd10,
               AVG(l.gd14) gd14, AVG(l.win) wr
        FROM laning l JOIN matches m ON l.match_id = m.match_id
        WHERE l.team_position = ? AND m.patch = ?
        GROUP BY l.champion HAVING n >= ?
        ORDER BY gd10 DESC
        """,
        (pos, patch, min_games),
    ).fetchall()
    print(f"=== {pos}：對線強度榜 (avg golddiff@10；patch {patch}；n>={min_games}) ===")
    print(f"  {'champ':<13}{'n':>4}{'gd@10':>8}{'gd@14':>8}{'cs@10':>7}{'winrate':>9}")
    for champ, n, gd10, xd10, cd10, gd14, wr in rows:
        print(f"  {champ:<13}{n:>4}{gd10:>8.0f}{gd14:>8.0f}{cd10:>7.1f}{wr * 100:>8.0f}%")
    print()


def main():
    conn = sqlite3.connect(config.DB_PATH)
    patches = conn.execute(
        "SELECT patch, COUNT(*) FROM matches GROUP BY patch ORDER BY 2 DESC"
    ).fetchall()
    n_m = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    print(f"dataset: {n_m} games; patches={dict(patches)}\n")

    patch = sys.argv[1] if len(sys.argv) > 1 else patches[0][0]
    min_games = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    for pos in ("TOP", "MIDDLE", "BOTTOM", "JUNGLE", "UTILITY"):
        table(conn, pos, patch, min_games)


if __name__ == "__main__":
    main()
