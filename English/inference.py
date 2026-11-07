"""English inference.

    python English/inference.py                          # all test sentences -> Results/English
    python English/inference.py --text "Hello there." --out hello.wav
"""

import argparse
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastspeech2.synthesis import Synthesizer, run_test_set  # noqa: E402
from fastspeech2.utils import load_config  # noqa: E402

HERE = Path(__file__).resolve().parent

# kept exactly as given in the assignment (including the broken word in E3)
TEST_SENTENCES = {
    "E1": "The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.",
    "E2a": "The orchestra's conduct during the ceremony was praised by the audience.",
    "E2b": "The researchers will conduct an experiment to evaluate the proposed method.",
    "E3": "The rhyth  m of the orchestra changed unexpectedly.",
    "E4": "The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--checkpoint", default=str(HERE / "checkpoints" / "best.pt"))
    parser.add_argument("--text", default=None)
    parser.add_argument("--out", default="output.wav")
    parser.add_argument("--speaker", default=None)
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
