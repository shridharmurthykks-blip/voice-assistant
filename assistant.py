import time
import wave
from collections import deque
import numpy as np
import sounddevice as sd
import ollama
from faster_whisper import WhisperModel
from piper import PiperVoice

print("Loading models...")
stt = WhisperModel("base", device="cpu", compute_type="int8")
tts = PiperVoice.load("en_US-lessac-medium.onnx")
messages = [{"role": "system", "content": "You are a helpful voice assistant. Keep answers short, 1 to 2 sentences."}]

def listen(max_seconds=15, silence_limit=1.0, threshold=0.003):
    print("\nListening...")
    chunk = 1600  # 0.1 s
    pre = deque(maxlen=3)
    frames, started, silent = [], False, 0
    with sd.InputStream(samplerate=16000, channels=1, dtype="float32", blocksize=chunk) as s:
        for _ in range(int(max_seconds / 0.1)):
            data, _ = s.read(chunk)
            loud = float(np.sqrt(np.mean(data ** 2))) > threshold
            if not started:
                pre.append(data)
                if loud:
                    started = True
                    frames.extend(pre)
            else:
                frames.append(data)
                silent = 0 if loud else silent + 1
                if silent >= silence_limit / 0.1:
                    break
    if not frames:
        return "", 0
    audio = np.concatenate(frames).flatten()
    t = time.time()
    segments, _ = stt.transcribe(audio, beam_size=1, vad_filter=True, language="en")
    text = " ".join(x.text for x in segments).strip()
    return text, time.time() - t

def speak(text):
    with wave.open("reply.wav", "wb") as f:
        tts.synthesize_wav(text, f)
    with wave.open("reply.wav", "rb") as f:
        rate = f.getframerate()
        data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
    sd.play(data, rate)
    sd.wait()

while True:
    text, stt_time = listen()
    if not text:
        continue
    print("You:", text)
    if "goodbye" in text.lower():
        speak("Goodbye!")
        break
    t1 = time.time()
    messages.append({"role": "user", "content": text})
    reply = ollama.chat(model="qwen2.5:3b", messages=messages)
    answer = reply["message"]["content"]
    messages.append({"role": "assistant", "content": answer})
    t2 = time.time()
    print("AI:", answer)
    speak(answer)
    print(f"Latency: STT {stt_time:.1f}s | LLM {t2-t1:.1f}s")