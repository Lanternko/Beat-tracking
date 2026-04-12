"""
Beat This! with DBN post-processing (dbn=True)
Ablation: tests whether adding DBN back helps or hurts.
Beat This! paper claims DBN is unnecessary — this verifies that claim.
"""
import os, sys, json, glob
import numpy as np
from beat_this.inference import File2Beats

AUDIO_DIR = './GTZAN/'
ANNOT_DIR = './GTZAN_annotations/beats'


def run(output_path):
    print("Beat This! + DBN post-processing")
    predictor = File2Beats(checkpoint_path='final0', device='cuda', dbn=True)
    print("Model ready.\n")

    results = {}
    audio_files = sorted(glob.glob(os.path.join(AUDIO_DIR, "**/*.wav"), recursive=True))
    processed = 0

    for audio_path in audio_files:
        file_id = os.path.basename(audio_path).replace('.wav', '').replace('.', '_')
        if not os.path.exists(os.path.join(ANNOT_DIR, f"gtzan_{file_id}.beats")):
            continue
        try:
            beats, _ = predictor(audio_path)
            results[file_id] = beats.tolist()
            processed += 1
            if processed % 200 == 0:
                print(f"  [{processed}] processed...")
        except Exception as e:
            print(f"Error {file_id}: {e}")

    with open(output_path, 'w') as f:
        json.dump(results, f)
    print(f"\nDone. {processed} files → {output_path}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "beat_this_dbn_predictions.json")
