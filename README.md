# Yuki - offline cpu based AI Terminal and Coder Agent

- STT whisper turbo model
- TTS mozilla tts
- LLM qwen2.5-coder:latest via ollama


## Security

The required argument is the project path, yuki will remain at this path and will not scape from it.
It will use relative paths always, there is also a path sanitizer.
But in theory could create a scape python file, so if you use the `go` mode to remove confirmations, be aware of what is doing just in case.

## Usage

- Normal mode with confirmations for executing commands:
```bash
    python3 yuki.py /home/sha0/src/someproject/ 
```

- No confirmations:
```bash
    python3 yuki.py /home/sha0/src/someproject/ go
```

- Read the previous context from ctx file
```bash
python3 yuki.py /home/sha0/src/someproject/ go ctx
```


## Features

- no gpu usage
- totally offline
- prompt and confirm via voice, also speak with it via voice. (stt)
- AI speaks with a nice voice.  (tts)
- only english language for now.
- sanitizer.
- code edition features, this ones dont need confirmation.
- go-ahead mode to launch commands with no confirmation

## Thanks

system prompt inspired on radare2/r2ai decai AI decompiler and agentic reverser



