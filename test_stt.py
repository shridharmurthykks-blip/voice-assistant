import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel

print("Loading model (first time downloads ~500 MB)...")
model = WhisperModel("small", device="cpu", compute_type="int8")

print("Speak now for 5 seconds...")
audio = sd.rec(int(5 * 16000), samplerate=16000, channels=1, dtype="float32")
sd.wait()

segments, info = model.transcribe(audio.flatten(), beam_size=1)
print("Language:", info.language)
print("You said:", " ".join(s.text for s in segments))