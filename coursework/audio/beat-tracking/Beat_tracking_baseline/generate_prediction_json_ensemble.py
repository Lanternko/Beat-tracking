"""
Beat Tracking: RNN + TCN Ensemble (madmom)
Reference: Böck et al. (2014), Davies & Böck (2019)

Ensemble strategy: average beat activations from RNNBeatProcessor and
TCNBeatProcessor, then decode with DBNBeatTrackingProcessor.

Usage:
    python generate_prediction_json_ensemble.py [output_path]
    Default output: ensemble_predictions.json
"""
import os, sys, json
import glob
import numpy as np
import madmom

AUDIO_DIR = './GTZAN/'
ANNOT_DIR = './GTZAN_annotations/beats'


def load_annotations(annot_path):
    data = np.loadtxt(annot_path)
    if data.ndim > 1:
        return data[:, 0]
    return data


def beat_tracking_to_json(output_path):
    print("Initializing RNN + TCN Ensemble processors...")
    rnn_proc = madmom.features.beats.RNNBeatProcessor()
    tcn_proc = madmom.features.beats.TCNBeatProcessor()
    dbn_proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
    print("Processors ready.\n")

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
            act_rnn = rnn_proc(audio_path)
            act_tcn = tcn_proc(audio_path)

            # Align lengths (both at fps=100, minor ±1 frame edge differences)
            min_len = min(len(act_rnn), len(act_tcn))
            act_ensemble = (act_rnn[:min_len] + act_tcn[:min_len]) / 2.0

            detections = dbn_proc(act_ensemble)
            results[file_id] = detections.tolist()
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
    output_path = sys.argv[1] if len(sys.argv) > 1 else "ensemble_predictions.json"
    beat_tracking_to_json(output_path)
