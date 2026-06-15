# Reproducing ProjektZero on Oracle's Elixir

ProjektZero-LoL-Model (AGPL-3.0, github.com/MRittinghouse/ProjektZero-LoL-Model) is a
pro-esports pre-game win predictor: Team-Elo + Player-Elo + TrueSkill + EGPM-dominance +
Side-EMA ensemble on Oracle's Elixir data. Last updated 2021, so it needs pandas-2.x fixes.

Results: see `../RESULTS.md` §1. Faithful ensemble **64.33%** (their 2021: 63.66%); honest
time-split **63.81%**, bootstrap 90% CI on accuracy **±0.9pp**, ECE 0.043.

## Recipe

```bash
# 1. deps + clone
pip install trueskill seaborn gdown          # plus numpy/pandas/scikit-learn/scipy
git clone https://github.com/MRittinghouse/ProjektZero-LoL-Model.git /tmp/projektzero

# 2. Oracle's Elixir data (S3 bucket in their code is dead; data now on Google Drive).
#    file IDs from github.com/HerrKurz/Esports_Data_Pipeline config.py:
mkdir -p /tmp/oe_data
gdown 1XXk2LO0CsNADBB1LRGOV5rUpyZdEZ8s2 -O /tmp/oe_data/2023_oe.csv   # 2023
gdown 1IjIEhLc9n8eLKeY-yh_YigKVWbhgGBsN -O /tmp/oe_data/2024_oe.csv   # 2024
gdown 1v6LRphp2kYciU4SXp0PCjEMuev1bDejc -O /tmp/oe_data/2025_oe.csv   # 2025
# (2021=1fzwTTz..., 2022=1EHmptHy..., older years in that config.py too)

# 3. two pandas-2.x patches to src/oracles_elixir.py (see below), then:
gdown 1hnpbrUpBMS1TZI7IovfpKeZfWJH1Aptm -O /tmp/oe_data/2026_oe.csv   # for forward-test

# 3. copy all scripts, then run (order matters: enrich before forward/strong)
cp *.py /tmp/projektzero/
cd /tmp/projektzero/src
python3 /tmp/projektzero/repro_driver.py      # faithful reproduction -> enriched_teams_2023_2025.csv
python3 /tmp/projektzero/ci_eval.py           # honest time-split + bootstrap 90% CI
python3 /tmp/projektzero/enrich_all.py        # re-enrich incl. 2026 -> enriched_teams_2023_2026.csv
python3 /tmp/projektzero/forward_test.py      # predict unseen tournaments (regional vs international)
python3 /tmp/projektzero/strong_vs_strong.py  # calibration by confidence + strong-vs-strong + Bo5
```

## Scripts
- `repro_driver.py` — faithful repro (full-data, pre-game ratings), ensemble 64.3%.
- `ci_eval.py` — honest time-split + bootstrap 90% CI (±0.9pp), ECE.
- `enrich_all.py` — re-enrich ratings on 2023-2026, keeps league/split/year (for forward-test).
- `forward_test.py` — train only on pre-event games, predict unseen tournaments. LCK 2026 road-to-MSI 69%, MSI/Worlds ~55%.
- `strong_vs_strong.py` — accuracy by model confidence (calibration), top-team subset, a real T1-vs-Gen.G Bo5 per-map.

## Patches to `src/oracles_elixir.py` (pandas 2.x)

1. `clean_data`, the dtype cast — bare `"datetime64"` is rejected in pandas 2.x:
   ```python
   # - "date": "datetime64",
   + "date": "datetime64[ns]",
   ```
2. `clean_data`, the player/team count filter — `value_counts().to_frame()` renames the
   count column to `count` (index becomes `gameid`) in pandas 2.x:
   ```python
   # - counts = oe_data["gameid"].value_counts().to_frame()
   # - counts = counts[(counts["gameid"] < cap) | (counts["gameid"] > cap)]
   + counts = oe_data["gameid"].value_counts()
   + counts = counts[(counts < cap) | (counts > cap)]
   ```

`repro_driver.py` also adds a data-quality filter (drop games lacking the 5 canonical
positions per side, or with a player listed twice — 12 academy games) that the original
pipeline assumed away; without it TrueSkill's positional parse expands the merge and crashes.

## Caveats
- Pro data, not soloQ — this is the **player/team strength** axis (project menu B/C), the
  counterpoint to draft. It is NOT comparable to the soloQ draft model in `../draft_winrate.py`.
- No betting-odds baseline, so "is 64% good?" for pro prediction is still open (markets are the
  real benchmark). The point here is the metric/calibration/CI harness + the strength-vs-draft contrast.
- Scripts hardcode `/tmp/projektzero` and `/tmp/oe_data`; adjust if you relocate.
