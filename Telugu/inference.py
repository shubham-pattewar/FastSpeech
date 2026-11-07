"""Telugu inference.

    python Telugu/inference.py                           # all test sentences -> Results/Telugu
    python Telugu/inference.py --text "నమస్కారం." --out namaskaram.wav
"""

import argparse
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastspeech2.synthesis import Synthesizer, run_test_set  # noqa: E402
from fastspeech2.utils import load_config  # noqa: E402

HERE = Path(__file__).resolve().parent

# T3 keeps the split word from the assignment on purpose
TEST_SENTENCES = {
    "T1": "న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు.",
    "T2a": "తాడుపై అతని పట్టు చాలా బలంగా ఉంది.",
    "T2b": "ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది.",
    "T3": "పులి అడ  విలో నిశ్శబ్దంగా నడిచింది.",
    "T4": "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి.",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--checkpoint", default=str(HERE / "checkpoints" / "best.pt"))
    parser.add_argument("--text", default=None)
    parser.add_argument("--out", default="output.wav")
    parser.add_argument("--speaker", default=None, help="female or male")
    parser.add_argument("--pitch_control", type=float, default=1.0)
    parser.add_argument("--energy_control", type=float, default=1.0)
    parser.add_argument("--duration_control", type=float, default=1.0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    synth = Synthesizer(args.checkpoint, cfg["path"]["vocoder"], args.device)
    controls = dict(pitch_control=args.pitch_control, energy_control=args.energy_control,
                    duration_control=args.duration_control)

    if args.text:
        res = synth(args.text, speaker=args.speaker, **controls)
        print(" ".join(res["phones"]))
        sf.write(args.out, res["wav"], synth.ap.sr)
        print(f"wrote {args.out}")
    else:
        run_test_set(synth, TEST_SENTENCES, cfg["path"]["results_dir"], speaker=args.speaker, **controls)


if __name__ == "__main__":
    main()
