import ollama

messages = []
while True:
    text = input("You: ")
    if text.lower() in ("quit", "exit"):
        break
    messages.append({"role": "user", "content": text})
    reply = ollama.chat(model="qwen2.5:3b", messages=messages)
    answer = reply["message"]["content"]
    messages.append({"role": "assistant", "content": answer})
    print("AI:", answer)