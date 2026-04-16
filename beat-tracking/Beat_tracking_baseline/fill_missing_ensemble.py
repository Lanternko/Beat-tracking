"""Fill missing predictions for Beat This! checkpoint ensemble."""
import os, json, glob
import numpy as np
from beat_this.inference import File2Beats

AUDIO_DIR = './GTZAN/'
ANNOT_DIR = './GTZAN_annotations/beats'
CHECKPOINTS = ['final0', 'final1', 'final2']
TOLERANCE = 0.070
THRESHOLD = 2

TARGET = 'checkpoint_ensemble_predictions.json'


def majority_vote(all_beats, tolerance=TOLERANCE, threshold=THRESHOLD):
    reference = all_beats[0]
    others = all_beats[1:]
    voted = []
    for t in reference:
        votes = 1
        for beats in others:
            if np.any(np.abs(beats - t) <= tolerance):
                votes += 1
        if votes >= threshold:
            voted.append(t)
    return np.array(voted)


def main():
    results = json.load(open(TARGET)) if os.path.exists(TARGET) else {}

    needed = []
    for ap in sorted(glob.glob(os.path.join(AUDIO_DIR, "**/*.wav"), recursive=True)):
        fid = os.path.basename(ap).replace('.wav', '').replace('.', '_')
        if fid in results:
            continue
        if not os.path.exists(os.path.join(ANNOT_DIR, f"gtzan_{fid}.beats")):
            continue
        needed.append((fid, ap))

    print(f"{TARGET}: existing {len(results)}, need {len(needed)}")
    if not needed:
        return

    print("Loading 3 checkpoints...")
    predictors = [File2Beats(checkpoint_path=c, device='cuda', dbn=False)
                  for c in CHECKPOINTS]

    for i, (fid, ap) in enumerate(needed):
        try:
            all_beats = [p(ap)[0] for p in predictors]
            results[fid] = majority_vote(all_beats).tolist()
            if (i+1) % 20 == 0 or i+1 == len(needed):
                print(f"  [{i+1}/{len(needed)}]")
        except Exception as e:
            print(f"  Error {fid}: {e}")

    with open(TARGET, 'w') as f:
        json.dump(results, f)
    print(f"Saved {TARGET}: {len(results)} entries")


if __name__ == "__main__":
    main()
