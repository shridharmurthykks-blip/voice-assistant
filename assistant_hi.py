import re
import time
import wave
import queue
import threading
from collections import deque
import numpy as np
import sounddevice as sd
import ollama
from faster_whisper import WhisperModel
from piper import PiperVoice

print("Loading models...")
stt = WhisperModel("small", device="cpu", compute_type="int8")
voices = {
    "en": PiperVoice.load("en_US-lessac-medium.onnx"),
    "hi": PiperVoice.load("hi_IN-pratham-medium.onnx"),
}

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. "
    "Reply in the same language the user speaks (Hindi or English). "
    "Keep answers short, 1 to 3 sentences. "
    "If replying in Hindi, use Devanagari script. "
    "Use simple, correct Hindi sentences."
)
messages = [{"role": "system", "content": SYSTEM_PROMPT}]


def listen(max_seconds=15, silence_limit=1.0, threshold=0.003):
    print("\nListening...")
    chunk = 1600  # 0.1 second
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
        return "", "en", 0
    audio = np.concatenate(frames).flatten()
    t = time.time()
    lang_id, prob, _ = stt.detect_language(audio)
    lang = "hi" if (lang_id == "hi" and prob > 0.6) else "en"
    segments, info = stt.transcribe(audio, beam_size=1, vad_filter=True, language=lang)
    text = " ".join(x.text for x in segments).strip()
    return text, lang, time.time() - t


def synth(text, lang):
    with wave.open("tmp.wav", "wb") as f:
        voices[lang].synthesize_wav(text, f)
    with wave.open("tmp.wav", "rb") as f:
        rate = f.getframerate()
        data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
    return data, rate


play_q = queue.Queue()


def player():
    while True:
        item = play_q.get()
        sd.play(*item)
        sd.wait()
        play_q.task_done()


threading.Thread(target=player, daemon=True).start()

SENTENCE_END = r"(.+?[.!?।])\s"

while True:
    text, lang, stt_time = listen()
    if not text:
        continue
    print(f"You ({lang}):", text)
    if "goodbye" in text.lower() or "अलविदा" in text:
        play_q.put(synth("Goodbye!", "en"))
        play_q.join()
        break

    hint = "Reply in English only." if lang == "en" else "Reply in Hindi only, in Devanagari script."
    messages.append({"role": "user", "content": text + "\n(" + hint + ")"})
    t1 = time.time()
    full, buf, first = "", "", None
    print("AI: ", end="", flush=True)
    for part in ollama.chat(model="qwen2.5:3b", messages=messages, stream=True):
        piece = part["message"]["content"]
        print(piece, end="", flush=True)
        full += piece
        buf += piece
        m = re.search(SENTENCE_END, buf)
        while m:
            play_q.put(synth(m.group(1), lang))
            buf = buf[m.end():]
            if first is None:
                first = time.time() - t1
            m = re.search(SENTENCE_END, buf)
    if buf.strip():
        play_q.put(synth(buf.strip(), lang))
        if first is None:
            first = time.time() - t1
    print()
    messages.append({"role": "assistant", "content": full})
    play_q.join()
    print(f"Latency: STT {stt_time:.1f}s | first speech {first:.1f}s")