import re
import io
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

LLM = {"en": "gemma2:2b", "hi": "llama3.2:3b"}
OPTS = {
    "en": {"num_ctx": 1024, "num_predict": 80, "temperature": 0.4},
    "hi": {"num_ctx": 1024, "num_predict": 60, "temperature": 0.3},
}
EMOJI = re.compile("[\U0001F000-\U0001FFFF\u2600-\u27BF\uFE0F\u200d]")

print("Loading models...")
stt_fast = WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=4)
stt_hi = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=4)
voices = {
    "en": PiperVoice.load("en_US-lessac-medium.onnx"),
    "hi": PiperVoice.load("hi_IN-priyamvada-medium.onnx"),
}

SYSTEM = (
    "You are Asha, a friendly female AI voice assistant running on the user's laptop. "
    "Answer the question directly. Never repeat the question. "
    "Reply ONLY in English or Hindi, in the language of the user's latest message. "
    "Keep replies to 1 or 2 short sentences. Do not use emojis. "
    "Write Hindi in Devanagari script, use simple everyday words and feminine forms "
    "such as 'मैं बता सकती हूँ'."
)
HINT = {
    "en": "Reply in English.",
    "hi": "Reply in simple Hindi (Devanagari) in one short sentence, under 12 words.",
}
history = []


def synth(text, lang):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        voices[lang].synthesize_wav(text, f)
    buf.seek(0)
    with wave.open(buf, "rb") as f:
        rate = f.getframerate()
        data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
    return data, rate


def run_stt(model, audio, language, prompt=None):
    segs, _ = model.transcribe(
        audio, beam_size=1, vad_filter=False, language=language,
        condition_on_previous_text=False, without_timestamps=True,
        initial_prompt=prompt,
    )
    return " ".join(s.text for s in segs).strip()


def understand(audio):
    lang_id, prob, _ = stt_fast.detect_language(audio)
    if lang_id == "hi" and prob >= 0.7:
        text = run_stt(stt_hi, audio, "hi", prompt="नमस्ते, आप कैसे हैं? मुझे बताइए।")
        return text, "hi"
    return run_stt(stt_fast, audio, "en"), "en"


def listen(max_seconds=15, silence_limit=0.6, threshold=0.003):
    print("\nListening...")
    chunk = 1600
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
    return np.concatenate(frames).flatten() if frames else None


def take_chunk(buf, early):
    m = re.search(r"(.+?[.!?।])\s", buf)
    if m:
        return m.group(1), buf[m.end():]
    if early:
        m = re.search(r"(.{12,}?[,;:])\s", buf)
        if m:
            return m.group(1), buf[m.end():]
        parts = buf.split(" ")
        if len(parts) >= 7:
            return " ".join(parts[:6]), " ".join(parts[6:])
    return None, buf


play_q = queue.Queue()


def player():
    while True:
        data, rate = play_q.get()
        sd.play(data, rate)
        sd.wait()
        play_q.task_done()


threading.Thread(target=player, daemon=True).start()

print("Warming up...")
silence = np.zeros(16000, dtype=np.float32)
stt_fast.detect_language(silence)
run_stt(stt_fast, silence, "en")
run_stt(stt_hi, silence, "hi")
for lg in ("en", "hi"):
    for _ in ollama.chat(model=LLM[lg],
                         messages=[{"role": "system", "content": SYSTEM},
                                   {"role": "user", "content": "Hello\n(" + HINT[lg] + ")"}],
                         stream=True, keep_alive="60m",
                         options={**OPTS[lg], "num_predict": 5}):
        pass
synth("Hello.", "en")
synth("नमस्ते।", "hi")
print("Ready. Say goodbye to quit.")

while True:
    audio = listen()
    if audio is None:
        continue
    t0 = time.time()
    text, lang = understand(audio)
    stt_time = time.time() - t0
    if not text:
        continue
    print(f"You ({lang}): {text}")
    words = re.findall(r"\w+", text.lower())
    if "goodbye" in words or "bye" in words or "अलविदा" in text:
        play_q.put(synth("Goodbye!", "en"))
        play_q.join()
        break

    history.append({"role": "user", "content": f"{text}\n({HINT[lang]})"})
    msgs = [{"role": "system", "content": SYSTEM}] + history[-4:]

    t1 = time.time()
    full, buf, first, llm_first, tts_first = "", "", None, None, 0.0
    print("AI: ", end="", flush=True)
    for part in ollama.chat(model=LLM[lang], messages=msgs, stream=True,
                            keep_alive="60m", options=OPTS[lang]):
        piece = EMOJI.sub("", part["message"]["content"])
        if llm_first is None:
            llm_first = time.time() - t1
        print(piece, end="", flush=True)
        full += piece
        buf += piece
        chunk, buf = take_chunk(buf, first is None)
        while chunk:
            ts = time.time()
            audio_out = synth(chunk, lang)
            if first is None:
                tts_first = time.time() - ts
            play_q.put(audio_out)
            if first is None:
                first = time.time() - t0
            chunk, buf = take_chunk(buf, False)
    if buf.strip():
        ts = time.time()
        audio_out = synth(buf.strip(), lang)
        if first is None:
            tts_first = time.time() - ts
        play_q.put(audio_out)
        if first is None:
            first = time.time() - t0
    print()
    history.append({"role": "assistant", "content": full})
    play_q.join()
    print(f"Latency: STT {stt_time:.1f}s | LLM first token {llm_first:.1f}s | "
          f"TTS first chunk {tts_first:.1f}s | first sound {first:.1f}s")