# Qwen3-TTS

Qwen3-TTS is an open-source series of text-to-speech models developed by the Qwen team ([Qwen3-TTS on Hugging Face](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)). This launcher gives you a 1-click install + browser UI for all three model types:

- **VoiceDesign** (1.7B) — create a custom voice from a natural-language description
- **Base** (0.6B / 1.7B) — voice cloning from a reference audio sample
- **CustomVoice** (0.6B / 1.7B) — TTS with a set of predefined speakers

## Hardware Support

The app automatically picks the best device:

| Hardware | Device | Notes |
| --- | --- | --- |
| NVIDIA GPU | CUDA | bf16, flash-attn optional |
| Apple Silicon | MPS (`app_mac.py`) | Whisper transcription runs on CPU |
| AMD GPU (Windows) | DirectML (`app_amd.py`) | fp32, whisper on CPU |
| CPU only | CPU | slow fallback |

### AMD (DirectML) notes

On Windows + AMD GPUs (`torch-directml`), generation is fully supported but a DirectML quirk requires a small compatibility shim installed by the launcher (`app_amd.py`). In practice this means:

- Models run in fp32 (DirectML has limited bf16 kernel support), so VRAM usage is roughly double what the CUDA build needs. Only the most recently used model is kept loaded to avoid exhausting VRAM when switching tabs.
- Whisper transcription runs on CPU.
- Generation is slower than CUDA, but fully GPU accelerated.

## Usage

1. **Install** — run the Install script. This clones the app, creates a virtualenv, installs Python dependencies, and installs the correct PyTorch build for your hardware.
2. **Start** — run the Start script. The web UI opens automatically.
3. Download a model on the **Models** tab (or let a generation tab download it on first use).
4. Pick a tab (Voice Design / Voice Clone / Custom Voice), enter your text, and generate.

## Getting a Custom Voice with the API

All UI functions are exposed as Gradio API endpoints. The examples below assume the app is running at `http://127.0.0.1:7860`.

### Python

```python
from gradio_client import Client, handle_file

client = Client("http://127.0.0.1:7860")

# Voice clone (uses model_type "Base" / "CustomVoice", model_size "0.6B" or "1.7B")
audio = client.predict(
    handle_file("reference.wav"),   # reference speaker audio
    "reference transcript",         # ref_text
    "Text to synthesize",           # target text
    "English",                      # language
    False,                          # use x-vector only (no ref transcript)
    "0.6B",                         # model size
    500,                            # max chars per chunk
    0,                              # chunk gap (seconds)
    -1,                             # seed (-1 = random)
    api_name="/generate_voice_clone",
)
audio_file, status = audio
```

### curl

```bash
# 1. Start a job (returns an event id)
EVENT_ID=$(curl -s -X POST http://127.0.0.1:7860/gradio_api/call/generate_custom_voice \
  -H "Content-Type: application/json" \
  -d '{"data": ["Hello world", "English", "Ryan", null, "1.7B", -1]}' | python -c "import sys,json; print(json.load(sys.stdin)['event_id'])")

# 2. Stream the result (SSE)
curl -N http://127.0.0.1:7860/gradio_api/call/generate_custom_voice/$EVENT_ID
```

### JavaScript

```js
import { Client } from "@gradio/client";

const client = await Client.connect("http://127.0.0.1:7860");
const audio = await client.predict("/generate_custom_voice", {
  text: "Hello world",
  language: "English",
  speaker: "Ryan",
  instruct: null,
  model_size: "1.7B",
  seed: -1,
});
```

### Available endpoints

| API name | Inputs (in order) |
| --- | --- |
| `/generate_voice_design` | `text`, `language`, `voice_description`, `seed` |
| `/generate_voice_clone` | `ref_audio` (file), `ref_text`, `target_text`, `language`, `use_xvector_only`, `model_size`, `max_chunk_chars`, `chunk_gap`, `seed` |
| `/generate_custom_voice` | `text`, `language`, `speaker`, `instruct`, `model_size`, `seed` |
| `/download_model` | `model_type`, `model_size` |
| `/get_available_sizes` | `model_type` |
| `/get_downloaded_models` | — |

## Troubleshooting

- **Logs** — installation and runtime logs for this launcher live in `logs/api` under this folder.
- **Reset** — if the install is corrupted, run the Reset script (deletes `app/venv`) then Install again.
- **VRAM** — on AMD, only one model is kept in memory; if you still run out of memory, use the 0.6B models or unload the model on the Models tab.