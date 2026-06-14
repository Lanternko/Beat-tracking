"""
Generate Beat This! predictions for files NOT yet in a given JSON.
Used to incrementally fill in missing predictions (e.g. newly uploaded rock files).
"""
import os, sys, json, glob
from beat_this.inference import File2Beats

AUDIO_DIR = './GTZAN/'
ANNOT_DIR = './GTZAN_annotations/beats'


def main(target_json, dbn_flag=False):
    if os.path.exists(target_json):
        results = json.load(open(target_json))
    else:
        results = {}

    audio_files = sorted(glob.glob(os.path.join(AUDIO_DIR, "**/*.wav"), recursive=True))
    needed = []
    for ap in audio_files:
        fid = os.path.basename(ap).replace('.wav', '').replace('.', '_')
        if fid in results:
            continue
        if not os.path.exists(os.path.join(ANNOT_DIR, f"gtzan_{fid}.beats")):
            continue
        needed.append((fid, ap))

    print(f"Loaded existing {target_json} with {len(results)} entries.")
    print(f"Need to generate {len(needed)} missing predictions (dbn={dbn_flag}).")

    if not needed:
        print("Nothing to do.")
        return

    predictor = File2Beats(checkpoint_path='final0', device='cuda', dbn=dbn_flag)
    print("Model loaded.\n")

    for i, (fid, ap) in enumerate(needed):
        try:
            beats, _ = predictor(ap)
            results[fid] = beats.tolist()
            if (i+1) % 20 == 0 or i+1 == len(needed):
                print(f"  [{i+1}/{len(needed)}] processed")
        except Exception as e:
            print(f"  Error {fid}: {e}")

    with open(target_json, 'w') as f:
        json.dump(results, f)
    print(f"\nSaved updated {target_json} with {len(results)} entries.")


if __name__ == "__main__":
    target = sys.argv[1]
    dbn = '--dbn' in sys.argv
    main(target, dbn_flag=dbn)
