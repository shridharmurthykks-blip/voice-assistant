import wave
import sounddevice as sd
import numpy as np
from piper import PiperVoice

voice = PiperVoice.load("en_US-lessac-medium.onnx")

with wave.open("out.wav", "wb") as f:
    voice.synthesize_wav("Hello Shridhar, I am your voice assistant.", f)

with wave.open("out.wav", "rb") as f:
    rate = f.getframerate()
    data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)

sd.play(data, rate)
sd.wait()
print("Done")