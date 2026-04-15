"""
Beat Tracking: RNN + CRF (madmom)
Compare CRFBeatDetectionProcessor vs DBNBeatTrackingProcessor as decoder.

CRF uses a discriminative model for beat sequence decoding,
vs DBN which uses a generative HMM-style approach.

Reference: Krebs et al., "Inferring Beat Rhythms Using a Multi-Scale Analysis
of Musical Audio", 2015.
"""
import os, sys, json, glob
import numpy as np
import madmom

AUDIO_DIR = './GTZAN/'
ANNOT_DIR = './GTZAN_annotations/beats'


def run(output_path):
    print("RNN + CRF Beat Detector")
    rnn_proc = madmom.features.beats.RNNBeatProcessor()
    crf_proc = madmom.features.beats.CRFBeatDetectionProcessor(fps=100)
    print("Processors ready.\n")

    results = {}
    audio_files = sorted(glob.glob(os.path.join(AUDIO_DIR, "**/*.wav"), recursive=True))
    processed = 0

    for audio_path in audio_files:
        file_id = os.path.basename(audio_path).replace('.wav', '').replace('.', '_')
        if not os.path.exists(os.path.join(ANNOT_DIR, f"gtzan_{file_id}.beats")):
            continue
        try:
            act = rnn_proc(audio_path)
            beats = crf_proc(act)
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
    run(sys.argv[1] if len(sys.argv) > 1 else "crf_predictions.json")
