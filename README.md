\# Voice AI Assistant (Fully Local, Open-Weight Models)



A voice assistant that listens, understands speech, replies with an LLM and speaks back. All models run locally. No paid or hosted inference APIs.



Architecture diagram: see architecture.md



\## Pipeline

Mic -> VAD -> STT (faster-whisper) -> LLM (Ollama qwen2.5:3b, streaming) -> TTS (Piper) -> Speaker



\## Models

\- STT: faster-whisper (base for English, small for Hindi/English)

\- LLM: qwen2.5:3b via Ollama

\- TTS: Piper (en\_US-lessac-medium, hi\_IN-pratham-medium)

\- VAD: energy-based detection plus Whisper VAD filter



\## Features

\- Context kept across turns

\- Voice activity detection

\- Streaming LLM output into TTS

\- Hindi and English with language detection

\- Web interface (FastAPI)



\## Hardware

Lenovo Yoga Slim 7, Intel Core Ultra 5 125H, 16 GB RAM, Intel Arc integrated graphics, Windows 11. CPU only.



\## Observed latency

\- small model, non-streaming: STT 2.7s, LLM 1.4s

\- base model, non-streaming: STT 0.6s, LLM 1.2s

\- base model, streaming: STT 0.7s, first speech 1.8s

\- Hindi (small, streaming): STT 3.3s, first speech 2.8s

\- Web app, first request: STT 1.6s, LLM 9.1s, TTS 6.4s



\## Setup

1\. Install Python 3.10+, Git and Ollama

2\. ollama pull qwen2.5:3b

3\. python -m venv venv, then venv\\Scripts\\activate

4\. pip install faster-whisper piper-tts fastapi uvicorn sounddevice numpy ollama python-multipart

5\. python -m piper.download\_voices en\_US-lessac-medium

6\. python -m piper.download\_voices hi\_IN-pratham-medium



\## Run

\- English streaming: python assistant\_stream.py

\- Hindi + English: python assistant\_hi.py

\- Web: uvicorn web\_app:app --port 8000, open http://localhost:8000

\- Say "goodbye" to stop.



\## Demo video

(add link here)



\## Known limitations

\- The 3B model is weaker in Hindi.

\- Language detection can mislabel short or accented English.

\- Latency depends on free RAM (CPU only).

