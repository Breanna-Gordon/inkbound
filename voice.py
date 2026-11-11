import io
import json
import wave
from pathlib import Path

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Whisper (OpenAI), run locally through Hugging Face transformers. English-only models are smaller and
# more accurate for English. "openai/whisper-small.en" is more accurate but slower.
MODEL_ID = "openai/whisper-base.en"
RATE = 16000  # Whisper listens at 16,000 samples a second
CLIPS = Path("data/voice")  # spoken moves with what was meant, for evaluate_voice.py
USE_HINT = True  # give Whisper the game's names and words to expect (evaluate_voice.py measures both ways)

_models = {}


def load(model_id=MODEL_ID):
    """Load a Whisper model the first time it's needed, then keep it."""
    if model_id not in _models:
        processor = WhisperProcessor.from_pretrained(model_id)
        model = WhisperForConditionalGeneration.from_pretrained(model_id)
        model.eval()
        _models[model_id] = (processor, model)
    return _models[model_id]


def read_wav(data):
    """WAV file bytes -> mono samples between -1 and 1, at 16 kHz."""
    with wave.open(io.BytesIO(data)) as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128
    else:
        dtype = {2: np.int16, 4: np.int32}[width]
        audio = np.frombuffer(raw, dtype=dtype).astype(np.float32) / np.iinfo(dtype).max
    audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != RATE and len(audio):  # resample by straight-line interpolation
        new_length = int(len(audio) * RATE / rate)
        audio = np.interp(np.linspace(0, len(audio) - 1, new_length), np.arange(len(audio)), audio)
    return audio.astype(np.float32)


def hint_text(words):
    """A short prompt for Whisper listing the names and words it should expect, e.g.
    "Thornwood Edge, Pip, tiefling". Whisper treats it as text that came just before, so it spells those
    words the way the game does."""
    names = []
    for w in words:
        w = (w or "").split(",")[0].strip()  # "Pip, the miller's daughter" -> "Pip"
        if w and not w.lower().startswith("the ") or w.startswith("The "):  # ordinary words are heard fine
            names.append(w)
    return ", ".join(dict.fromkeys(names))[:600]  # well under Whisper's limit for a prompt


def game_words():
    """Every place, character, race, class and skill name in the game."""
    import rules  # only needed here
    words = []
    for path in sorted(Path("locations").glob("*.json")):
        location = json.loads(path.read_text(encoding="utf-8"))
        words += [location["name"], *location["cast"]]
    return words + list(rules.RACES) + list(rules.CLASSES) + list(rules.SKILLS)


def transcribe(data, model_id=MODEL_ID, hint=None):
    """WAV bytes -> the words Whisper heard. hint: names and words to expect (see hint_text)."""
    audio = read_wav(data)
    if len(audio) < RATE * 0.3 or np.abs(audio).max() < 0.01:  # too short or silent
        return ""
    processor, model = load(model_id)
    features = processor(audio, sampling_rate=RATE, return_tensors="pt").input_features
    extra = {}
    if hint:
        try:
            extra["prompt_ids"] = processor.get_prompt_ids(hint, return_tensors="pt")
        except Exception:  # an older or different processor: listen without the hint
            extra = {}
    with torch.no_grad():
        try:
            ids = model.generate(features, max_new_tokens=96, **extra)
        except Exception:  # if this model can't take a hint, listen without one
            if not extra:
                raise
            ids = model.generate(features, max_new_tokens=96)
    text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    if hint and text.startswith(hint):  # some versions give back the hint too
        text = text[len(hint):].strip()
    return text


def save_clip(data, heard, meant, model_id=MODEL_ID, source="play"):
    """Keep a spoken move with what Whisper heard and what the player meant (after any corrections).
    source: "play" (said during a game) or "practice" (read out in record_voice.py)."""
    CLIPS.mkdir(parents=True, exist_ok=True)
    number = len(list(CLIPS.glob("*.wav"))) + 1
    (CLIPS / f"{number:04d}.wav").write_bytes(data)
    (CLIPS / f"{number:04d}.json").write_text(
        json.dumps({"heard": heard, "meant": meant, "model": model_id, "source": source}, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "test_speech.wav"
    print("Heard:", transcribe(Path(path).read_bytes()))
