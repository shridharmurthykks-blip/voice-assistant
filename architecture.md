\# Architecture



```mermaid

flowchart LR

&#x20;   A\[Microphone] --> B\[VAD<br/>stops listening on silence]

&#x20;   B --> C\[STT<br/>faster-whisper]

&#x20;   C -->|text + language| D\[LLM<br/>qwen2.5:3b via Ollama<br/>keeps chat history]

&#x20;   D -->|streamed sentence by sentence| E\[TTS<br/>Piper English / Hindi]

&#x20;   E --> F\[Speaker]

```



All models run locally on CPU. No hosted inference APIs are used.

