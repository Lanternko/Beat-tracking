"""
Beat This! Checkpoint Ensemble (final0 ~ final4)

Strategy: run all 5 checkpoints independently, then for each file
apply majority vote on beat positions (tolerance = 70ms).
A beat time is kept if >= 3/5 checkpoints agree within the window.

This mirrors the ensemble approach described in Böck et al. (2024).
"""
import os, sys, json, glob
import numpy as np
from beat_this.inference import File2Beats

AUDIO_DIR = './GTZAN/'
ANNOT_DIR = './GTZAN_annotations/beats'
CHECKPOINTS = ['final0', 'final1', 'final2']
TOLERANCE = 0.070   # 70ms
THRESHOLD = 2       # majority = at least 2/3


def majority_vote(all_beats, tolerance=TOLERANCE, threshold=THRESHOLD):
    """
    Keep beat times that are confirmed by >= threshold checkpoints.
    Uses final0 as the reference grid and counts votes from others.
    """
    reference = all_beats[0]
    others = all_beats[1:]
    voted = []
    for t in reference:
        votes = 1  # reference itself counts
        for beats in others:
            if np.any(np.abs(beats - t) <= tolerance):
                votes += 1
        if votes >= threshold:
            voted.append(t)
    return np.array(voted)


def run(output_path):
    print(f"Loading {len(CHECKPOINTS)} checkpoints...")
    predictors = [File2Beats(checkpoint_path=c, device='cuda', dbn=False)
                  for c in CHECKPOINTS]
    print("All checkpoints ready.\n")

    results = {}
    audio_files = sorted(glob.glob(os.path.join(AUDIO_DIR, "**/*.wav"), recursive=True))
    processed = 0

    for audio_path in audio_files:
        file_id = os.path.basename(audio_path).replace('.wav', '').replace('.', '_')
        if not os.path.exists(os.path.join(ANNOT_DIR, f"gtzan_{file_id}.beats")):
            continue
        try:
            all_beats = [p(audio_path)[0] for p in predictors]
            ensemble_beats = majority_vote(all_beats)
            results[file_id] = ensemble_beats.tolist()
            processed += 1
            if processed % 200 == 0:
                print(f"  [{processed}] processed...")
        except Exception as e:
            print(f"Error {file_id}: {e}")

    with open(output_path, 'w') as f:
        json.dump(results, f)
    print(f"\nDone. {processed} files → {output_path}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "checkpoint_ensemble_predictions.json")
