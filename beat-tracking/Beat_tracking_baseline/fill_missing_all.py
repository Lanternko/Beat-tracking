"""
Fill in missing predictions for ALL methods.
Computes RNN activation once per missing file, then applies all RNN-based methods.
Also updates Beat This! checkpoint ensemble.
"""
import os, sys, json, glob
import numpy as np
import madmom

AUDIO_DIR = './GTZAN/'
ANNOT_DIR = './GTZAN_annotations/beats'

# Per-genre best DBN params (from dbn_per_genre.py earlier run)
PER_GENRE_BEST = {
    'blues':     (200, 32),
    'classical': ( 50, 16),
    'country':   (100, 16),
    'disco':     (150, 24),
    'hiphop':    (150, 16),
    'jazz':      (200,  8),
    'metal':     ( 50, 32),
    'pop':       (300, 16),
    'reggae':    (200,  8),
    'rock':      (100, 24),
}

# Literature genre-aware params (from genre_aware_dbn.py)
LITERATURE_PARAMS = {
    'blues':     (100, 16),
    'country':   (100, 16),
    'disco':     (150, 16),
    'hiphop':    (150, 16),
    'metal':     (150,  8),
    'pop':       (100, 16),
    'reggae':    (100,  8),
    'rock':      (100, 16),
    'classical': ( 30,  8),
    'jazz':      ( 30,  8),
}


def genre_of(fid):
    return fid.rsplit('_', 1)[0]


def main():
    targets = {
        'crf_predictions.json':           'crf',
        'dbn_per_genre_predictions.json': 'per_genre',
        'genre_aware_predictions.json':   'literature',
        'baseline.json':                  'default',
    }

    # Find files needing predictions in any target
    audio_files = sorted(glob.glob(os.path.join(AUDIO_DIR, "**/*.wav"), recursive=True))
    needed_per_target = {p: set() for p in targets}

    existing = {p: json.load(open(p)) if os.path.exists(p) else {} for p in targets}
    for ap in audio_files:
        fid = os.path.basename(ap).replace('.wav', '').replace('.', '_')
        if not os.path.exists(os.path.join(ANNOT_DIR, f"gtzan_{fid}.beats")):
            continue
        for p in targets:
            if fid not in existing[p]:
                needed_per_target[p].add(fid)

    all_needed = set.union(*needed_per_target.values())
    print(f"Missing per target:")
    for p, files in needed_per_target.items():
        print(f"  {p}: {len(files)}")
    print(f"Total unique files needing RNN activation: {len(all_needed)}\n")

    if not all_needed:
        print("Nothing to do.")
        return

    # Build mapping from fid → audio_path
    audio_map = {}
    for ap in audio_files:
        fid = os.path.basename(ap).replace('.wav', '').replace('.', '_')
        audio_map[fid] = ap

    # Initialise processors
    rnn = madmom.features.beats.RNNBeatProcessor()
    crf = madmom.features.beats.CRFBeatDetectionProcessor(fps=100)
    dbn_default = madmom.features.beats.DBNBeatTrackingProcessor(
        fps=100, transition_lambda=100, observation_lambda=16)

    # Cache DBN procs by params
    dbn_cache = {}
    def get_dbn(tl, ol):
        key = (tl, ol)
        if key not in dbn_cache:
            dbn_cache[key] = madmom.features.beats.DBNBeatTrackingProcessor(
                fps=100, transition_lambda=tl, observation_lambda=ol)
        return dbn_cache[key]

    # Process each missing file
    for i, fid in enumerate(sorted(all_needed)):
        ap = audio_map.get(fid)
        if not ap:
            print(f"  Skip {fid}: audio not found")
            continue
        try:
            act = rnn(ap)
            genre = genre_of(fid)

            if fid in needed_per_target['baseline.json']:
                existing['baseline.json'][fid] = dbn_default(act).tolist()
            if fid in needed_per_target['crf_predictions.json']:
                existing['crf_predictions.json'][fid] = crf(act).tolist()
            if fid in needed_per_target['dbn_per_genre_predictions.json']:
                tl, ol = PER_GENRE_BEST.get(genre, (100, 16))
                existing['dbn_per_genre_predictions.json'][fid] = get_dbn(tl, ol)(act).tolist()
            if fid in needed_per_target['genre_aware_predictions.json']:
                tl, ol = LITERATURE_PARAMS.get(genre, (100, 16))
                existing['genre_aware_predictions.json'][fid] = get_dbn(tl, ol)(act).tolist()

            if (i+1) % 10 == 0 or i+1 == len(all_needed):
                print(f"  [{i+1}/{len(all_needed)}] {fid}")
        except Exception as e:
            print(f"  Error {fid}: {e}")

    # Save all
    for p in targets:
        with open(p, 'w') as f:
            json.dump(existing[p], f)
        print(f"  Saved {p}: {len(existing[p])} entries")


if __name__ == "__main__":
    main()
