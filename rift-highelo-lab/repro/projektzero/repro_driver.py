# Faithful reproduction of ProjektZero's ensemble on Oracle's Elixir 2023-2025.
# Replicates data_generator.enrich_dataset + model_validator, no CSV round-trip.
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, '/tmp/projektzero')
os.chdir('/tmp/projektzero/src')
import numpy as np
import pandas as pd
from pathlib import Path
import src.oracles_elixir as oe
import src.lol_modeling as lol
import src.model_validator as mv

files = ['/tmp/oe_data/2023_oe.csv', '/tmp/oe_data/2024_oe.csv', '/tmp/oe_data/2025_oe.csv']
data = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
print('raw rows', data.shape, '| games', data['gameid'].nunique())

invalid_games = ['NA1/3754345055', 'NA1/3754344502', 'ESPORTSTMNT02/1890835',
                 'NA1/3669212337', 'NA1/3669211958', 'ESPORTSTMNT02/1890848',
                 'ESPORTSTMNT02_1932895', 'ESPORTSTMNT02_1932914']
data = data[~data.gameid.isin(invalid_games)].copy()

teams = oe.clean_data(data, split_on='team')
players = oe.clean_data(data, split_on='player')
print('cleaned teams', teams.shape, '| players', players.shape)

# trueskill_model.setup_match assumes each game has the 5 canonical positions per
# side; messy minor-league rows (dup/missing position) break the positional parse.
def valid_gameids(pl):
    sig = pl.groupby(['gameid', 'side'])['position'].agg(lambda s: tuple(sorted(set(s))))
    good = sig[sig == ('bot', 'jng', 'mid', 'sup', 'top')].reset_index()
    cnt = good.groupby('gameid').size()
    pos_ok = set(cnt[cnt == 2].index)
    # also require 10 distinct playerids/game: some academy games list a player in
    # two positions (data error), which duplicates TrueSkill merge keys.
    npid = pl.groupby('gameid')['playerid'].nunique()
    pid_ok = set(npid[npid == 10].index)
    return pos_ok & pid_ok

vg = valid_gameids(players)
players = players[players.gameid.isin(vg)].reset_index(drop=True)
teams = teams[teams.gameid.isin(vg)].reset_index(drop=True)
print(f'after position-validity filter: {len(vg)} games | teams {teams.shape} players {players.shape}')

# Replicate enrich_dataset (minus CSV writes)
teams = lol.dk_enrich(teams, entity='team')
players = lol.dk_enrich(players, entity='player')
players = lol.player_elo(players)
teams = lol.team_elo(teams)
teams = lol.aggregate_player_elos(players, teams)
players, teams, ts_lookup = lol.trueskill_model(players, teams, initial_sigma=2.75)
teams = lol.egpm_model(teams, "team")
teams = lol.ewm_model(teams, "team")
teams = lol.enrich_ema_statistics(teams, "team")
print('enriched teams', teams.shape)

# NaN-safety: the 5 model win-prob columns must be present & non-null for scoring
wp_cols = ['team_elo_win_perc', 'player_elo_win_perc', 'trueskill_win_perc',
           'egpm_dom_logistic_win_perc', 'side_ema_win_perc']
before = len(teams)
teams = teams.dropna(subset=wp_cols).reset_index(drop=True)
print(f'dropped {before - len(teams)} rows with NaN win-prob; kept {len(teams)} team-rows '
      f'({teams.gameid.nunique()} games)')

directory = Path('/tmp/projektzero/reports/figures')
te = mv.validate_team_elo(teams, directory, False)
pe = mv.validate_player_elo(teams, directory, False)
ts = mv.validate_trueskill(teams, directory, False)
ed = mv.validate_egpm_dominance(teams, directory, False)
se = mv.validate_side_ema(teams, directory, False)
es = mv.validate_ensemble_accuracy(teams, te[0], pe[0], ts[0], ed[0], se[0], directory, False)

print('\n=== FAITHFUL REPRODUCTION (full-data, pre-game ratings) ===')
print(f"{'model':12s} {'acc':>8s} {'logloss':>9s} {'brier':>8s}   (ProjektZero 2021)")
ref = {'TeamElo': '61.79/.6498/.2290', 'PlayerElo': '62.60/.6423/.2257',
       'TrueSkill': '62.48/.6404/.2250', 'EGPM': '59.53/.6637/.2357',
       'SideEMA': '51.93/1.662/.2700', 'Ensemble': '63.66/.6425/.2255'}
for name, m in [('TeamElo', te), ('PlayerElo', pe), ('TrueSkill', ts),
                ('EGPM', ed), ('SideEMA', se), ('Ensemble', es)]:
    print(f"{name:12s} {m[0]*100:7.2f}% {m[1]:9.4f} {m[2]:8.4f}   [{ref[name]}]")

# persist enriched team data for the CI step
teams.to_csv('/tmp/oe_data/enriched_teams_2023_2025.csv', index=False)
print('\nsaved enriched team data -> /tmp/oe_data/enriched_teams_2023_2025.csv')
