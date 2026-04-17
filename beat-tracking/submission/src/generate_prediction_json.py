"""
Beat Tracking with Beat This! (Böck et al., 2024)
Reference: https://github.com/CPJKU/beat_this
Architecture: Conv + Time-Frequency Transformer (no DBN post-processing)
Reported GTZAN F1: ~89%

Usage:
    python generate_prediction_json.py [output_path]
    Default output: predictions.json
"""
import os, sys, json
import glob
import numpy as np
from beat_this.inference import File2Beats

AUDIO_DIR = './GTZAN/'
ANNOT_DIR = './GTZAN_annotations/beats'


def beat_tracking_to_json(output_path):
    print("Initializing Beat This! model...")
    print("(Will auto-download checkpoint ~50MB on first run)")

    # checkpoint='final0': best single-model checkpoint from Beat This! paper
    # dbn=False: Beat This! uses shift-tolerant post-processing instead of DBN
    predictor = File2Beats(checkpoint_path='final0', device='cuda', dbn=False)
    print("Model ready.\n")

    results = {}

    search_pattern = os.path.join(AUDIO_DIR, "**/*.wav")
    audio_files = sorted(glob.glob(search_pattern, recursive=True))
    total = len(audio_files)
    processed = 0

    for audio_path in audio_files:
        file_id = os.path.basename(audio_path).replace('.wav', '').replace('.', '_')

        annot_path = os.path.join(ANNOT_DIR, f"gtzan_{file_id}.beats")
        if not os.path.exists(annot_path):
            continue

        try:
            # Returns (beat_times_seconds, downbeat_times_seconds)
            beats, downbeats = predictor(audio_path)
            results[file_id] = beats.tolist()
            processed += 1

            if processed % 100 == 0:
                print(f"  [{processed}/{total}] processed...")

        except Exception as e:
            print(f"Error processing {file_id}: {e}")

    with open(output_path, 'w') as f:
        json.dump(results, f)

    print(f"\nDone. {processed} files processed.")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    output_path = sys.argv[1] if len(sys.argv) > 1 else "predictions.json"
    beat_tracking_to_json(output_path)
