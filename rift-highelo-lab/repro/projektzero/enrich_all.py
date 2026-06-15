# Re-enrich ProjektZero ratings on 2023-2026 (walk-forward), keep league+split+year,
# save for the 2026 forward-test.
import os, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, '/tmp/projektzero'); os.chdir('/tmp/projektzero/src')
import numpy as np, pandas as pd
import src.oracles_elixir as oe
import src.lol_modeling as lol

files = [f'/tmp/oe_data/{y}_oe.csv' for y in [2023, 2024, 2025, 2026]]
raw = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
invalid = ['NA1/3754345055', 'NA1/3754344502', 'ESPORTSTMNT02/1890835', 'NA1/3669212337',
           'NA1/3669211958', 'ESPORTSTMNT02/1890848', 'ESPORTSTMNT02_1932895', 'ESPORTSTMNT02_1932914']
raw = raw[~raw.gameid.isin(invalid)].copy()
split_map = raw.drop_duplicates('gameid').set_index('gameid')['split'].to_dict()

teams = oe.clean_data(raw, split_on='team')
players = oe.clean_data(raw, split_on='player')

def valid_gameids(pl):
    sig = pl.groupby(['gameid', 'side'])['position'].agg(lambda s: tuple(sorted(set(s))))
    good = sig[sig == ('bot', 'jng', 'mid', 'sup', 'top')].reset_index()
    cnt = good.groupby('gameid').size()
    pos_ok = set(cnt[cnt == 2].index)
    npid = pl.groupby('gameid')['playerid'].nunique()
    return pos_ok & set(npid[npid == 10].index)

vg = valid_gameids(players)
players = players[players.gameid.isin(vg)].reset_index(drop=True)
teams = teams[teams.gameid.isin(vg)].reset_index(drop=True)
print('games', len(vg))

teams = lol.dk_enrich(teams, 'team'); players = lol.dk_enrich(players, 'player')
players = lol.player_elo(players)
teams = lol.team_elo(teams)
teams = lol.aggregate_player_elos(players, teams)
players, teams, _ = lol.trueskill_model(players, teams, initial_sigma=2.75)
teams = lol.egpm_model(teams, "team")
teams = lol.ewm_model(teams, "team")
teams = lol.enrich_ema_statistics(teams, "team")

wp = ['team_elo_win_perc', 'player_elo_win_perc', 'trueskill_win_perc',
      'egpm_dom_logistic_win_perc', 'side_ema_win_perc']
teams = teams.dropna(subset=wp).reset_index(drop=True)
teams['split'] = teams.gameid.map(split_map)
teams.to_csv('/tmp/oe_data/enriched_teams_2023_2026.csv', index=False)
print('saved enriched_teams_2023_2026.csv |', len(teams), 'team-rows |', teams.gameid.nunique(), 'games')
print('2026 leagues:', sorted(teams[pd.to_datetime(teams.date).dt.year == 2026].league.unique()))
