# Talk — Voice-to-Text Hotkey App

## Problem
Build a global push-to-talk voice-to-text tool that captures speech, transcribes it using one of four selectable STT models, and pastes the result into the currently focused application. The UI shows a real-time waveform during recording and a processing indicator during transcription. Optionally, the transcribed text can be live-translated to a target language before pasting.

## Repository
**GitHub:** `crutkas/Talk`

## Approach
1. **Python prototype** — Full working app with all features
2. **C# port** — Polished Windows app with native system tray integration
3. **Rust port** — Lightweight, high-performance binary
4. **Translation layer** — Optional real-time translation pipeline (defaults to OFF)

All three share the same JSON config format and the same local model server architecture (where applicable).

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Global Hotkey Listener (background)            │
│  ┌───────────────────────────────────────────┐  │
│  │  Hold Key → Start Recording               │  │
│  │  Release Key → Stop Recording → Transcribe│  │
│  └───────────────────────────────────────────┘  │
│                                                 │
│  ┌───────────────────────────────────────────┐  │
│  │  Floating Overlay (near cursor)           │  │
│  │  ┌─────────────────────────────────────┐  │  │
│  │  │  [Waveform Visualizer]              │  │  │
│  │  │  [Model: Whisper v3 ▾]              │  │  │
│  │  │  [Status: Recording... / Processing]│  │  │
│  │  └─────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────┘  │
│                                                 │
│  ┌───────────────────────────────────────────┐  │
│  │  STT Engine Abstraction Layer             │  │
│  │  ├── WhisperEngine (in-process)           │  │
│  │  ├── CanaryQwenEngine (HTTP / NeMo)       │  │
│  │  ├── VoxtralEngine (HTTP / vLLM)          │  │
│  │  └── Qwen3ASREngine (in-process / HTTP)   │  │
│  └───────────────────────────────────────────┘  │
│                                                 │
│  ┌───────────────────────────────────────────┐  │
│  │  Translation Layer (optional, off by def) │  │
│  │  ├── NLLB-200 (text, 200 langs, light)    │  │
│  │  ├── SeamlessM4T v2 (speech+text, 100L)   │  │
│  │  └── Madlad-400 (text, 400 langs, edge)   │  │
│  └───────────────────────────────────────────┘  │
│                                                 │
│  ┌───────────────────────────────────────────┐  │
│  │  Paste Output → Clipboard + SendInput     │  │
│  └───────────────────────────────────────────┘  │
│                                                 │
│  System Tray Icon (settings, model toggle,      │
│  quit, config)                                  │
└─────────────────────────────────────────────────┘
```

## Config Format (config.json)

```json
{
  "hotkey": "ctrl+shift+space",
  "default_model": "whisper",
  "models": {
    "whisper": {
      "enabled": true,
      "mode": "in-process",
      "model_size": "large-v3-turbo",
      "device": "auto"
    },
    "canary_qwen": {
      "enabled": true,
      "mode": "http",
      "endpoint": "http://localhost:8001/transcribe"
    },
    "voxtral": {
      "enabled": true,
      "mode": "http",
      "endpoint": "http://localhost:8002/transcribe"
    },
    "qwen3_asr": {
      "enabled": true,
      "mode": "in-process",
      "model_name": "Qwen/Qwen3-ASR-1.7B"
    }
  },
  "audio": {
    "sample_rate": 16000,
    "channels": 1,
    "format": "wav"
  },
  "translation": {
    "enabled": false,
    "target_language": "es",
    "model": "nllb-200",
    "models": {
      "nllb-200": {
        "mode": "in-process",
        "model_name": "facebook/nllb-200-1.3B",
        "device": "auto"
      },
      "seamless-m4t": {
        "mode": "in-process",
        "model_name": "facebook/seamless-m4t-v2-large",
        "device": "auto"
      },
      "madlad-400": {
        "mode": "in-process",
        "model_name": "google/madlad400-3b-mt",
        "device": "auto"
      }
    }
  },
  "ui": {
    "overlay_width": 320,
    "overlay_height": 100,
    "waveform_color": "#4CAF50",
    "processing_color": "#FF9800",
    "opacity": 0.9
  }
}
```

## Phase 1: Python Prototype

### Tech Stack
- **Audio capture**: `sounddevice` (PortAudio binding)
- **Hotkey**: `pynput` (global keyboard listener)
- **UI/Overlay**: `PyQt6` (frameless floating widget with waveform)
- **Waveform rendering**: `numpy` + QPainter or `pyqtgraph`
- **STT engines**:
  - Whisper: `faster-whisper` (CTranslate2, fastest local inference)
  - Canary Qwen: `nemo_toolkit` or HTTP client
  - Voxtral: `transformers` + `vllm` or HTTP client
  - Qwen3-ASR: `qwen-asr` pip package
- **Clipboard/paste**: `pyperclip` + `pynput` keyboard controller
- **Config**: stdlib `json`
- **System tray**: `PyQt6` QSystemTrayIcon

### Todos

#### 1. Project scaffolding
- Create repo structure, pyproject.toml, requirements.txt
- Set up config.json with defaults
- Create README with setup instructions

#### 2. Audio capture module
- `sounddevice` stream with 16kHz mono PCM
- Ring buffer for waveform visualization (last ~2 seconds of samples)
- Start/stop recording on hotkey press/release
- Export captured audio as WAV bytes for STT

#### 3. STT engine abstraction
- Base class `STTEngine` with `transcribe(audio_bytes) -> str`
- `WhisperEngine` — loads `faster-whisper` model, transcribes in-process
- `CanaryQwenEngine` — HTTP POST to NeMo server endpoint
- `VoxtralEngine` — HTTP POST to vLLM/HF server endpoint
- `Qwen3ASREngine` — in-process via `qwen-asr` package
- Engine factory: instantiate engine by name from config

#### 4. Floating overlay UI
- PyQt6 frameless, always-on-top, translucent widget
- Appears near mouse cursor on hotkey press
- Real-time waveform: draws audio amplitude from ring buffer (~30fps)
- Three visual states:
  - **Recording** — green waveform animating
  - **Processing** — orange pulsing/spinner, waveform frozen
  - **Translating** — blue pulsing indicator, shows source text updating to translated text in real-time
  - **Done** — brief green flash, then auto-dismiss
- Shows current model name in small label
- When translation is active, overlay shows two lines:
  - Top: original transcription (dimmed)
  - Bottom: translated text (highlighted, updating in real-time as translation streams)
- Disappears after paste completes

#### 5. Paste output module
- Copy transcribed text to clipboard
- Simulate Ctrl+V keystroke to paste into focused app
- Small delay (~50ms) between clipboard set and paste to ensure reliability
- Restore previous clipboard contents after paste (optional, nice-to-have)

#### 6. Global hotkey manager
- `pynput` keyboard listener running in background thread
- Configurable hotkey combo from config.json
- Press = start recording + show overlay
- Release = stop recording + start transcription + processing indicator
- Debounce to prevent accidental double-triggers

#### 7. System tray integration
- PyQt6 QSystemTrayIcon with context menu:
  - Model selector (radio buttons for the 4 engines)
  - Translation toggle (on/off) with target language submenu
  - Translation model selector (NLLB-200 / SeamlessM4T v2 / Madlad-400)
  - Settings (opens config.json in editor)
  - Quit
- Tray tooltip shows current model + translation status

#### 8. End-to-end integration & testing
- Wire all modules together
- Test with each STT engine
- Handle errors gracefully (model not available, timeout, etc.)
- Add logging for debugging

## Phase 1b: Translation Layer

### Translation Model Recommendations

| Model | Languages | VRAM | Best For | License |
|---|---|---|---|---|
| **NLLB-200** (Meta) | 200+ | ~3GB | Broadest language coverage, low-resource languages | CC-BY-NC |
| **SeamlessM4T v2** (Meta) | 100+ | ~6GB | Highest quality for common languages, speech+text | CC-BY-NC |
| **Madlad-400** (Google) | 400+ | ~4GB | Maximum language count, edge-friendly | Apache 2.0 |

**Links:**
- NLLB-200: https://huggingface.co/facebook/nllb-200-1.3B
- SeamlessM4T v2: https://huggingface.co/facebook/seamless-m4t-v2-large
- Madlad-400: https://huggingface.co/google/madlad400-3b-mt

### Tech Stack (Python)
- **NLLB-200**: `transformers` + `ctranslate2` (for speed)
- **SeamlessM4T v2**: `transformers` (seamless_communication)
- **Madlad-400**: `transformers`

### Todos

#### 9. Translation engine abstraction
- Base class `TranslationEngine` with `translate(text: str, source_lang: str, target_lang: str) -> str`
- `NLLBEngine` — loads NLLB-200 1.3B via CTranslate2 for fast inference
- `SeamlessEngine` — loads SeamlessM4T v2 large for highest quality
- `MadladEngine` — loads Madlad-400 3B for maximum language coverage
- Engine factory: instantiate by name from config
- All engines support streaming/partial output for real-time UI updates

#### 10. Translation pipeline integration
- After STT produces text, if translation enabled:
  - Feed English text to selected translation engine
  - Stream translated tokens back to overlay UI in real-time
  - Overlay shows original (dimmed) + translated (highlighted) text
  - On completion, paste the translated text (not the original)
- Translation runs async so overlay updates progressively
- Handles edge cases: empty transcription, unsupported language pair

#### 11. Translation UI updates
- Update overlay to show dual-line display when translation is active
- Add "Translating..." state with blue indicator
- Show target language flag/code in overlay
- System tray: translation toggle, target language picker, translation model selector

## Phase 2: C# Port (WPF/.NET 8)

### Tech Stack
- **Audio capture**: NAudio
- **Hotkey**: Win32 `RegisterHotKey` via P/Invoke
- **UI/Overlay**: WPF frameless window with `WriteableBitmap` waveform
- **STT**: HTTP client to model servers (all 4 models run as Python servers)
- **Paste**: Win32 `SendInput` via P/Invoke
- **System tray**: `Hardcodet.NotifyIcon.Wpf`
- **Config**: `System.Text.Json`

### Todos

#### 12. C# project scaffolding
- .NET 8 WPF app, single-file publish
- Shared config.json format (same as Python)
- NuGet packages: NAudio, Hardcodet.NotifyIcon.Wpf

#### 13. Audio capture (NAudio)
- WaveInEvent with 16kHz mono
- Ring buffer for waveform data
- WAV byte export

#### 14. STT client layer
- `ISTTEngine` interface with `Task<string> TranscribeAsync(byte[] audio)`
- Four HTTP client implementations posting to local model servers
- Model server launcher helper (starts Python servers if not running)

#### 15. Translation client layer
- `ITranslationEngine` interface with `Task<string> TranslateAsync(string text, string sourceLang, string targetLang)`
- HTTP client to Python translation server
- Streaming support for real-time overlay updates

#### 16. WPF floating overlay
- Frameless, always-on-top, per-pixel transparent WPF window
- Waveform rendered via WriteableBitmap or SkiaSharp
- Same three visual states as Python version
- Position near cursor using `GetCursorPos`

#### 17. Hotkey & paste (Win32)
- RegisterHotKey / UnregisterHotKey P/Invoke
- SendInput for Ctrl+V paste
- Clipboard via `System.Windows.Clipboard`

#### 18. System tray & settings
- NotifyIcon with context menu
- Model selector, settings, quit

## Phase 3: Rust Port

### Tech Stack
- **Audio capture**: `cpal`
- **Hotkey**: `winapi` / `windows` crate (`RegisterHotKey`)
- **UI/Overlay**: `egui` with `eframe` (or `iced`) — lightweight GPU-rendered
- **STT**: `reqwest` HTTP client to model servers, or `whisper-rs` (whisper.cpp bindings) for in-process Whisper
- **Paste**: `winapi` `SendInput` + `clipboard-win`
- **System tray**: `tray-icon` crate
- **Config**: `serde_json`

### Todos

#### 19. Rust project scaffolding
- Cargo workspace, dependencies
- Shared config.json deserialization via serde

#### 20. Audio capture (cpal)
- Input stream at 16kHz mono
- Ring buffer, WAV encoding via `hound`

#### 21. STT client layer
- Trait `SttEngine` with `fn transcribe(&self, audio: &[u8]) -> Result<String>`
- `WhisperRsEngine` — in-process via `whisper-rs` (whisper.cpp)
- HTTP client engines for Canary Qwen, Voxtral, Qwen3-ASR

#### 22. Translation client layer
- Trait `TranslationEngine` with `fn translate(&self, text: &str, src: &str, tgt: &str) -> Result<String>`
- HTTP client to Python translation server
- Streaming support via Server-Sent Events

#### 23. egui floating overlay
- Frameless, always-on-top via `eframe` window hints
- Waveform plot via `egui::plot` or custom painter
- Same three visual states

#### 24. Hotkey & paste (Win32)
- RegisterHotKey via `windows` crate
- SendInput + clipboard via `clipboard-win`

#### 25. System tray
- `tray-icon` with model toggle menu

## Repo Structure

```
Talk/                # github.com/crutkas/Talk
├── README.md
├── config.json                    # Shared config format
├── python/
│   ├── pyproject.toml
│   ├── requirements.txt
│   ├── src/
│   │   ├── __init__.py
│   │   ├── main.py                # Entry point
│   │   ├── audio.py               # Mic capture + ring buffer
│   │   ├── engines/
│   │   │   ├── __init__.py
│   │   │   ├── base.py            # STTEngine base class
│   │   │   ├── whisper_engine.py
│   │   │   ├── canary_engine.py
│   │   │   ├── voxtral_engine.py
│   │   │   └── qwen3_engine.py
│   │   ├── translation/
│   │   │   ├── __init__.py
│   │   │   ├── base.py            # TranslationEngine base class
│   │   │   ├── nllb_engine.py
│   │   │   ├── seamless_engine.py
│   │   │   └── madlad_engine.py
│   │   ├── hotkey.py              # Global hotkey manager
│   │   ├── overlay.py             # PyQt6 floating waveform UI
│   │   ├── paste.py               # Clipboard + SendKeys
│   │   └── tray.py                # System tray icon
│   └── tests/
├── csharp/
│   ├── VoxHotkey.sln
│   ├── VoxHotkey/
│   │   ├── VoxHotkey.csproj
│   │   ├── App.xaml
│   │   ├── MainWindow.xaml        # Floating overlay
│   │   ├── Audio/
│   │   ├── Engines/
│   │   ├── Translation/
│   │   ├── Hotkey/
│   │   └── Tray/
│   └── VoxHotkey.Tests/
├── rust/
│   ├── Cargo.toml
│   ├── src/
│   │   ├── main.rs
│   │   ├── audio.rs
│   │   ├── engines/
│   │   ├── translation/
│   │   ├── hotkey.rs
│   │   ├── overlay.rs
│   │   └── tray.rs
│   └── tests/
└── servers/                       # Model server scripts (for C#/Rust)
    ├── serve_canary.py
    ├── serve_voxtral.py
    ├── serve_qwen3asr.py
    ├── serve_translation.py       # Translation model server (NLLB/Seamless/Madlad)
    └── requirements-servers.txt
```

## Key Design Decisions

1. **Shared config.json** — All three implementations read the same config format, making migration seamless
2. **STT abstraction** — Clean interface in all languages so adding new models is trivial
3. **Python models stay Python** — For C# and Rust ports, the STT models run as local HTTP servers (in `servers/`). Only Whisper gets native bindings (whisper.cpp for Rust)
4. **Overlay near cursor** — Appears where you're looking, non-intrusive
5. **Push-to-talk** — Most natural for dictation; hold to record, release to transcribe
6. **Translation defaults to OFF** — When enabled, adds a translation step between STT and paste. Overlay shows real-time dual-line display (original + translated). Only the translated text is pasted.

## Hardware Target
- Primary: AMD Ryzen AI Max 395 with 128GB unified memory
- Models can run in-process (Python) or as local servers
- Full bf16 models fit comfortably in memory

## Dependencies to Install First
- Python 3.11+, PyQt6, sounddevice, pynput, faster-whisper, numpy
- Translation: transformers, ctranslate2, sentencepiece
- .NET 8 SDK (for C# phase)
- Rust toolchain (for Rust phase)
- Ollama (optional, for easy model serving)
