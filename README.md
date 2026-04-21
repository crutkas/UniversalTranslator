# Talk

A global push-to-talk voice-to-text tool with optional real-time translation. Press a hotkey, speak, and the transcribed (and optionally translated) text is pasted into whatever app has focus.

## Features

- **Push-to-talk** — Hold hotkey to record, release to transcribe
- **4 STT engines** — Whisper, Canary Qwen 2.5B, Voxtral Transcribe 2, Qwen3-ASR
- **3 translation engines** — NLLB-200, SeamlessM4T v2, Madlad-400 (optional, defaults off)
- **Floating overlay** — Shows real-time waveform near cursor with status indicators
- **Auto-paste** — Transcribed text is pasted into the focused app via Ctrl+V
- **System tray** — Model selection, translation toggle, target language picker
- **Configurable** — JSON config for hotkey, models, UI appearance

## Quick Start

### Python Prototype

```bash
cd python
pip install -e ".[dev]"
python -m src.main
```

### Configuration

Edit `config.json` in the project root:

```json
{
  "hotkey": "ctrl+shift+space",
  "default_model": "whisper",
  "translation": {
    "enabled": false,
    "target_language": "es",
    "model": "nllb-200"
  }
}
```

### Running Tests

```bash
cd python
pytest tests/ -v
```

### Linting

```bash
cd python
ruff check src/ tests/
ruff format src/ tests/
```

## Architecture

```
Hotkey Press → Start Recording → Show Overlay (waveform)
Hotkey Release → Stop Recording → STT Engine → [Translation] → Paste
```

### STT Models

| Engine | Type | Notes |
|--------|------|-------|
| Whisper (faster-whisper) | In-process | Default, requires `faster-whisper` |
| Canary Qwen 2.5B | HTTP | Requires NeMo server |
| Voxtral Transcribe 2 | HTTP | Requires vLLM server |
| Qwen3-ASR | In-process | Requires `qwen-asr` |

### Translation Models

| Engine | Languages | Notes |
|--------|-----------|-------|
| NLLB-200 | 200+ | Best coverage for low-resource languages |
| SeamlessM4T v2 | 100+ | Highest quality for common languages |
| Madlad-400 | 400+ | Maximum language count |

## Project Structure

```
Talk/
├── config.json              # Shared config
├── python/                  # Python prototype
│   ├── src/
│   │   ├── main.py          # Entry point
│   │   ├── audio.py         # Mic capture + ring buffer
│   │   ├── hotkey.py        # Global hotkey manager
│   │   ├── overlay.py       # Floating waveform UI
│   │   ├── paste.py         # Clipboard + paste
│   │   ├── tray.py          # System tray
│   │   ├── config.py        # Config loader
│   │   ├── engines/         # STT engines
│   │   └── translation/     # Translation engines
│   └── tests/
├── csharp/                  # C# port (planned)
├── rust/                    # Rust port (planned)
└── servers/                 # Model server scripts
```

## Hardware Target

Optimized for AMD Ryzen AI Max 395 with 128GB unified memory, but works on any machine with sufficient VRAM for the selected models.

## License

MIT
