# Voice AI Assistant (Fully Local, Open-Weight Models)

A voice assistant that listens, understands speech, replies with an LLM and speaks back. All models run locally. No paid or hosted inference APIs.

Architecture diagram: see [architecture.md](architecture.md)

## Pipeline

Mic -> VAD -> STT (faster-whisper) -> LLM (Ollama qwen2.5:3b, streaming) -> TTS (Piper) -> Speaker

## Models

- STT: faster-whisper (base for English, small for Hindi/English)
- LLM: qwen2.5:3b via Ollama
- TTS: Piper (en_US-lessac-medium, hi_IN-pratham-medium)
- VAD: energy-based detection plus Whisper VAD filter

## Features

- Context kept across turns
- Voice activity detection
- Streaming LLM output into TTS
- Hindi and English with language detection
- Web interface (FastAPI)

## Hardware

Lenovo Yoga Slim 7, Intel Core Ultra 5 125H, 16 GB RAM, Intel Arc integrated
## Demo video

https://drive.google.com/file/d/11tS0xHNMZ2licWXGLILRLx_a2Q_rr3k1/view?usp=sharing
