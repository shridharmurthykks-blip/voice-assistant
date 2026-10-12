import io
import time
import wave
import numpy as np
import ollama
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, Response
from faster_whisper import WhisperModel
from piper import PiperVoice

app = FastAPI()
print("Loading models...")
stt = WhisperModel("base", device="cpu", compute_type="int8")
tts = PiperVoice.load("en_US-lessac-medium.onnx")
messages = [{"role": "system", "content": "You are a helpful voice assistant. Keep answers short, 1 to 2 sentences."}]

@app.get("/", response_class=HTMLResponse)
def home():
    return open("index.html", encoding="utf-8").read()

@app.post("/talk")
async def talk(file: UploadFile = File(...)):
    raw = await file.read()
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    t0 = time.time()
    segments, _ = stt.transcribe(audio, beam_size=1, vad_filter=True, language="en")
    text = " ".join(s.text for s in segments).strip()
    t1 = time.time()
    if not text:
        return Response(status_code=204)
    messages.append({"role": "user", "content": text})
    reply = ollama.chat(model="qwen2.5:3b", messages=messages)
    answer = reply["message"]["content"]
    messages.append({"role": "assistant", "content": answer})
    t2 = time.time()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        tts.synthesize_wav(answer, f)
    t3 = time.time()
    print(f"You: {text}\nAI: {answer}\nSTT {t1-t0:.1f}s | LLM {t2-t1:.1f}s | TTS {t3-t2:.1f}s")
    headers = {
        "X-You": text.encode("ascii", "ignore").decode(),
        "X-AI": answer.replace("\n", " ").encode("ascii", "ignore").decode(),
        "X-Latency": f"STT {t1-t0:.1f}s | LLM {t2-t1:.1f}s | TTS {t3-t2:.1f}s",
    }
    return Response(buf.getvalue(), media_type="audio/wav", headers=headers)