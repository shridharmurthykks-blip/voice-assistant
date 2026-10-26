# Voice AI Assistant (Fully Local, Open-Weight Models)

A voice assistant that listens, understands speech, replies with an LLM and speaks back in English or Hindi. All models run locally on CPU. No paid or hosted inference APIs.

Architecture diagram: see [architecture.md](architecture.md)

## Pipeline

Mic -> VAD -> STT (faster-whisper) -> LLM (Ollama, streaming) -> TTS (Piper) -> Speaker

The LLM reply is streamed, and speech starts after the first few words, before the full reply is written.

## Models

- STT: faster-whisper (base for English, small for Hindi)
- LLM: gemma2:2b for English, llama3.2:3b for Hindi, via Ollama
- TTS: Piper, en_US-lessac-medium and hi_IN-priyamvada-medium (both female, so the voice stays the same when the language changes)
- VAD: energy-based detection

## Features

- Context kept across turns
- Voice activity detection
- Streaming LLM output into TTS
- Hindi and English with automatic language detection
- Same voice character in both languages
- Web interface (FastAPI)

## Hardware

Lenovo Yoga Slim 7, Intel Core Ultra 5 125H, 16 GB RAM, Intel Arc integrated graphics, Windows 11. CPU only.

## Observed latency

Time from end of speech to first sound:

| Turn | STT | LLM first token | First sound |
|---|---|---|---|
| English | 0.8s | 1.0s | 2.2s |
| Hindi | 1.8s | 0.8s | 3.5s |
| Hindi | 1.8s | 1.3s | 4.5s |
| English | 0.8s | 1.6s | 2.9s |

An earlier version without these optimisations took up to 8.9s to first sound.

How latency was reduced:
- All models are warmed up at startup and the LLM stays loaded
- One quick language check, so only Hindi uses the larger Whisper model
- Small LLMs with short replies
- No disk writes for audio
- Speech starts after the first few words while the reply is still streaming

## Setup

1. Install Python 3.10+, Git and Ollama
2. `ollama pull gemma2:2b`
3. `ollama pull llama3.2:3b`
4. `python -m venv venv`, then `venv\Scripts\activate`
5. `pip install faster-whisper piper-tts fastapi uvicorn sounddevice numpy ollama python-multipart`
6. `python -m piper.download_voices en_US-lessac-medium`
7. `python -m piper.download_voices hi_IN-priyamvada-medium`

## Run

- Main version (English and Hindi): `python assistant_v2.py`
- Web interface: `uvicorn web_app:app --port 8000`, then open http://localhost:8000
- Say "bye" or "goodbye" to stop.

## Demo video

https://drive.google.com/file/d/1tgHYPPOl7N5Xxf_-3or3cx0mne8RLfYG/view?usp=drivesdk

## Known limitations

- Hindi answer quality is limited by the small local model. A larger model would be better but slower.
- Language detection can mislabel short or accented English.
- CPU only, so latency depends on free RAM and power mode.
