import argparse
import csv
import json
import re
import time
from pathlib import Path

import voice

RESULTS = Path("results")
MODELS = ["openai/whisper-tiny.en", "openai/whisper-base.en", "openai/whisper-small.en"]


def words(text):
    """Lower case, no punctuation, split into words, so 'Grab it!' and 'grab it' count as the same."""
    return re.sub(r"[^a-z0-9' ]+", " ", text.lower()).split()


def word_errors(heard, meant):
    """How many words must be changed, added or removed to turn what was heard into what was meant
    (the edit distance used for word error rate)."""
    h, m = words(heard), words(meant)
    row = list(range(len(m) + 1))
    for i, hw in enumerate(h, start=1):
        previous, row[0] = row[0], i
        for j, mw in enumerate(m, start=1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (hw != mw))
    return row[len(m)]


def load_clips(source="all"):
    """Every saved spoken move: (wav bytes, what the player meant, clip name). source: play, practice or all."""
    clips = []
    for meta in sorted(voice.CLIPS.glob("*.json")):
        info = json.loads(meta.read_text(encoding="utf-8"))
        wav = meta.with_suffix(".wav")
        if source != "all" and info.get("source", "play") != source:
            continue
        if wav.exists() and info.get("meant", "").strip():
            clips.append((wav.read_bytes(), info["meant"], meta.stem))
    return clips


def main():
    p = argparse.ArgumentParser(description="How well each Whisper model hears spoken moves, with and without the "
                                            "game-vocabulary hint: word error rate and speed, on data/voice/.")
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--source", default="all", choices=["all", "play", "practice"],
                   help="moves said while playing, read out in record_voice.py, or both")
    args = p.parse_args()
    clips = load_clips(args.source)
    if not clips:
        print("No spoken moves saved yet. Play using the microphone, or run: streamlit run record_voice.py")
        return
    total_words = sum(len(words(meant)) for _, meant, _ in clips)
    hint = voice.hint_text(voice.game_words())
    print(f"{len(clips)} spoken moves ({args.source}), {total_words} words")

    rows = [("model", "vocabulary hint", "clips", "word error rate %", "moves heard perfectly %",
             "seconds per move")]
    mistakes = []
    for model_id in args.models:
        voice.load(model_id)  # load first, so loading time isn't counted as listening time
        for use_hint in (False, True):
            errors = perfect = 0
            start = time.time()
            for data, meant, name in clips:
                heard = voice.transcribe(data, model_id, hint if use_hint else None)
                e = word_errors(heard, meant)
                errors += e
                perfect += e == 0
                if e:
                    mistakes.append(f"{model_id} {'with' if use_hint else 'without'} hint · {name}: "
                                    f"heard \"{heard}\" / meant \"{meant}\"")
            seconds = (time.time() - start) / len(clips)
            wer = errors / max(1, total_words)
            label = "yes" if use_hint else "no"
            rows.append((model_id, label, len(clips), f"{wer * 100:.1f}", f"{perfect / len(clips) * 100:.1f}",
                         f"{seconds:.2f}"))
            print(f"  {model_id:26s} hint {label:3s}  word error rate {wer:6.1%}  "
                  f"heard perfectly {perfect / len(clips):6.1%}  {seconds:.2f} s per move")

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "voice_evaluation.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    (RESULTS / "voice_mistakes.txt").write_text("\n".join(mistakes), encoding="utf-8")
    print("\nSaved results/voice_evaluation.csv and results/voice_mistakes.txt")


if __name__ == "__main__":
    main()
