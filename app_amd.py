# coding=utf-8
import os
import sys
import warnings

# Suppress common warnings
warnings.filterwarnings("ignore", message=".*Min value of input waveform.*")
warnings.filterwarnings("ignore", message=".*Max value of input waveform.*")
warnings.filterwarnings("ignore", message=".*Trying to convert audio automatically.*")
warnings.filterwarnings("ignore", message=".*pad_token_id.*")
warnings.filterwarnings("ignore", message=".*Setting `pad_token_id`.*")

# Flash-attn is installed via torch.js during Pinokio install - no runtime install needed

import gradio as gr
import numpy as np
import torch
from huggingface_hub import snapshot_download, scan_cache_dir

# Windows consoles default to cp1252 when stdout is piped/captured; the app's
# emoji print() calls then raise UnicodeEncodeError and kill generation jobs.
# Force UTF-8 so captured output always works (Pinokio captures stdout).
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# DirectML support (AMD GPUs on Windows). This file is run only when torch.js
# installs torch-directml, but keep the import guarded so the app still runs
# (falling back to CPU) if the DML install somehow did not complete.
try:
    import torch_directml as dml
except Exception:
    dml = None

DML_DEVICE = None
DML_PATCHED = False


def get_device():
    """Return 'cuda', 'dml', or 'cpu' depending on what is available."""
    if torch.cuda.is_available():
        return "cuda"
    if dml is not None:
        return "dml"
    return "cpu"


def _install_dml_compat():
    """Patch torch so transformers generation works on DirectML.

    DirectML rejects a few operation shapes that CUDA accepts happily:
    - empty (0-element) tensors and some int64 ops used by transformers
      generation fail with "RuntimeError: The parameter is incorrect".
    - `torch.inference_mode()` produces tensors whose version counter cannot be
      set by DirectML ops ("Cannot set version_counter for inference tensor").
    These shims run the offending small tensors on CPU and move the result back
    to the DML device. Big fp32 tensors (model forward pass, KV cache) stay on
    the GPU.
    """
    global DML_PATCHED
    if DML_PATCHED:
        return
    torch.inference_mode = torch.no_grad

    _orig_gather = torch.gather
    _orig_scatter = torch.scatter
    _orig_scatter_ = torch.Tensor.scatter_
    _orig_cat = torch.cat

    def _dml_gather(input, dim, index, sparse_grad=False, out=None):
        if input.device.type == "privateuseone":
            r = _orig_gather(input.cpu(), dim, index.cpu() if index is not None else None, sparse_grad=sparse_grad)
            if out is None:
                return r.to(input.device)
            out.copy_(r.to(input.device))
            return out
        return _orig_gather(input, dim, index, sparse_grad=sparse_grad, out=out)

    def _dml_scatter(input, dim, index, src):
        if input.device.type == "privateuseone":
            idx = index.cpu() if index is not None else None
            src_c = src.cpu() if isinstance(src, torch.Tensor) else src
            return _orig_scatter(input.cpu(), dim, idx, src_c).to(input.device)
        return _orig_scatter(input, dim, index, src)

    def _dml_scatter_(input, dim, index, src):
        if input.device.type == "privateuseone":
            idx = index.cpu() if index is not None else None
            src_c = src.cpu() if isinstance(src, torch.Tensor) else src
            input_cpu = input.cpu().clone()
            _orig_scatter_(input_cpu, dim, idx, src_c)
            input.copy_(input_cpu)
            return input
        return _orig_scatter_(input, dim, index, src)

    def _dml_cat(tensors, dim=0, *, out=None):
        if any(isinstance(t, torch.Tensor) and t.device.type == "privateuseone" and (t.numel() == 0 or t.dtype == torch.int64) for t in tensors if isinstance(t, torch.Tensor)):
            tt = [(t.cpu() if isinstance(t, torch.Tensor) and t.device.type == "privateuseone" else t) for t in tensors]
            r = _orig_cat(tt, dim, out=None)
            if out is None:
                return r.to(tensors[0].device)
            out.copy_(r)
            return out
        return _orig_cat(tensors, dim, out=out)

    torch.gather = _dml_gather
    torch.scatter = _dml_scatter
    torch.Tensor.scatter_ = _dml_scatter_
    torch.cat = _dml_cat
    DML_PATCHED = True


if dml is not None:
    _install_dml_compat()

# Whisper model for transcription
whisper_model = None


def get_whisper_model():
    """Load Whisper tiny model for transcription."""
    global whisper_model
    if whisper_model is None:
        import whisper
        whisper_model = whisper.load_model("tiny", device="cuda" if get_device() == "cuda" else "cpu")
    return whisper_model


def unload_whisper():
    """Force unload whisper model from GPU."""
    global whisper_model
    if whisper_model is not None:
        # Move to CPU first, then delete
        try:
            whisper_model.cpu()
        except:
            pass
        whisper_model = None
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def transcribe_audio(audio):
    """Transcribe audio using Whisper tiny."""
    global whisper_model
    if audio is None:
        return "Please upload audio first."
    
    try:
        sr, wav = audio
        # Convert to float32 and normalize properly
        wav = wav.astype(np.float32)
        
        # Check if audio needs normalization (int16 range is -32768 to 32767)
        max_val = np.abs(wav).max()
        if max_val > 1.0:
            wav = wav / max_val  # Normalize to [-1, 1] range
        
        # Whisper expects 16kHz mono
        if len(wav.shape) > 1:
            wav = wav.mean(axis=1)
        
        if sr != 16000:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        
        model = get_whisper_model()
        result = model.transcribe(wav, fp16=False)
        text = result["text"].strip()
        
        # Unload whisper to free GPU memory
        unload_whisper()
        
        return text
    except Exception as e:
        # Still try to unload on error
        unload_whisper()
        return f"Transcription error: {str(e)}"

# Global model holders - keyed by (model_type, model_size)
loaded_models = {}

# Model size options
MODEL_SIZES = ["0.6B", "1.7B"]

# Available models configuration
AVAILABLE_MODELS = {
    "VoiceDesign": {
        "sizes": ["1.7B"],
        "description": "Create custom voices using natural language descriptions"
    },
    "Base": {
        "sizes": ["0.6B", "1.7B"],
        "description": "Voice cloning from reference audio"
    },
    "CustomVoice": {
        "sizes": ["0.6B", "1.7B"],
        "description": "TTS with predefined speakers and style instructions"
    }
}


def get_model_repo_id(model_type: str, model_size: str) -> str:
    """Get HuggingFace repo ID for a model."""
    return f"Qwen/Qwen3-TTS-12Hz-{model_size}-{model_type}"


def get_model_path(model_type: str, model_size: str) -> str:
    """Get model path based on type and size."""
    return snapshot_download(get_model_repo_id(model_type, model_size))


def check_model_downloaded(model_type: str, model_size: str) -> bool:
    """Check if a model is already downloaded in the cache."""
    try:
        cache_info = scan_cache_dir()
        repo_id = get_model_repo_id(model_type, model_size)
        for repo in cache_info.repos:
            if repo.repo_id == repo_id:
                return True
        return False
    except Exception:
        return False


def get_downloaded_models_status() -> str:
    """Get status of all available models."""
    lines = ["### Model Download Status\n"]
    for model_type, info in AVAILABLE_MODELS.items():
        lines.append(f"**{model_type}** - {info['description']}")
        for size in info["sizes"]:
            status = "✅ Downloaded" if check_model_downloaded(model_type, size) else "⬜ Not downloaded"
            lines.append(f"  - {size}: {status}")
        lines.append("")
    return "\n".join(lines)


def download_model(model_type: str, model_size: str, progress=gr.Progress()):
    """Download a specific model."""
    if model_size not in AVAILABLE_MODELS.get(model_type, {}).get("sizes", []):
        return f"❌ Invalid combination: {model_type} {model_size}", get_downloaded_models_status()
    
    repo_id = get_model_repo_id(model_type, model_size)
    
    if check_model_downloaded(model_type, model_size):
        return f"✅ {model_type} {model_size} is already downloaded!", get_downloaded_models_status()
    
    try:
        progress(0, desc=f"Downloading {model_type} {model_size}...")
        snapshot_download(repo_id)
        progress(1, desc="Complete!")
        return f"✅ Successfully downloaded {model_type} {model_size}!", get_downloaded_models_status()
    except Exception as e:
        return f"❌ Error downloading {model_type} {model_size}: {str(e)}", get_downloaded_models_status()


def get_available_sizes(model_type: str):
    """Get available sizes for a model type."""
    return gr.update(choices=AVAILABLE_MODELS.get(model_type, {}).get("sizes", []), value=AVAILABLE_MODELS.get(model_type, {}).get("sizes", ["1.7B"])[0])


def get_model(model_type: str, model_size: str):
    """Get or load a model by type and size."""
    global loaded_models, DML_DEVICE
    key = (model_type, model_size)
    if key not in loaded_models:
        from qwen_tts import Qwen3TTSModel
        model_path = get_model_path(model_type, model_size)
        dev = get_device()
        if dev == "cuda":
            loaded_models[key] = Qwen3TTSModel.from_pretrained(
                model_path,
                device_map="cuda",
                dtype=torch.bfloat16,
#           attn_implementation="flash_attention_2",
            )
        elif dev == "dml":
            # DirectML: device_map cannot be used (it maps to cuda:0, and
            # safetensors cannot load onto a privateuseone device), and bf16 is
            # not broadly supported by DirectML kernels, so load fp32 on CPU and
            # move the weights onto the DML device afterward.
            # fp32 weights are ~2x CUDA's bf16, so keep only the most recently
            # used model loaded to avoid exhausting VRAM when switching tabs.
            if DML_DEVICE is None:
                DML_DEVICE = dml.device()
            for other_key in list(loaded_models.keys()):
                if other_key != key:
                    del loaded_models[other_key]
                    print(f"[amd] evicted {other_key} to free DirectML VRAM")
            tts = Qwen3TTSModel.from_pretrained(model_path, dtype=torch.float32)
            tts.model = tts.model.to(DML_DEVICE)
            tts.device = DML_DEVICE
            loaded_models[key] = tts
        else:
            loaded_models[key] = Qwen3TTSModel.from_pretrained(
                model_path,
                device_map="cpu",
                dtype=torch.float32,
            )
    return loaded_models[key]


def get_loaded_models_status() -> str:
    """Get status of currently loaded models in memory."""
    if not loaded_models:
        return "No models currently loaded in memory."
    
    lines = ["**Currently loaded models:**"]
    for (model_type, model_size) in loaded_models.keys():
        lines.append(f"- {model_type} ({model_size})")
    return "\n".join(lines)


def load_model_manual(model_type: str, model_size: str, progress=gr.Progress()):
    """Manually load a model into memory."""
    if model_size not in AVAILABLE_MODELS.get(model_type, {}).get("sizes", []):
        return f"❌ Invalid combination: {model_type} {model_size}", get_loaded_models_status()
    
    key = (model_type, model_size)
    if key in loaded_models:
        return f"✅ {model_type} {model_size} is already loaded!", get_loaded_models_status()
    
    try:
        progress(0, desc=f"Loading {model_type} {model_size}...")
        get_model(model_type, model_size)
        progress(1, desc="Complete!")
        return f"✅ Successfully loaded {model_type} {model_size}!", get_loaded_models_status()
    except Exception as e:
        return f"❌ Error loading {model_type} {model_size}: {str(e)}", get_loaded_models_status()


def unload_model(model_type: str, model_size: str):
    """Unload a specific model from memory."""
    global loaded_models
    key = (model_type, model_size)
    
    if key not in loaded_models:
        return f"⚠️ {model_type} {model_size} is not loaded.", get_loaded_models_status()
    
    try:
        del loaded_models[key]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return f"✅ Unloaded {model_type} {model_size} and freed GPU memory.", get_loaded_models_status()
    except Exception as e:
        return f"❌ Error unloading: {str(e)}", get_loaded_models_status()


def unload_all_models():
    """Unload all models from memory."""
    global loaded_models
    
    if not loaded_models:
        return "⚠️ No models are currently loaded.", get_loaded_models_status()
    
    try:
        count = len(loaded_models)
        loaded_models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return f"✅ Unloaded {count} model(s) and freed GPU memory.", get_loaded_models_status()
    except Exception as e:
        return f"❌ Error unloading: {str(e)}", get_loaded_models_status()


def _normalize_audio(wav, eps=1e-12, clip=True):
    """Normalize audio to float32 in [-1, 1] range."""
    x = np.asarray(wav)

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6:
            y = y / (m + eps)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")

    if clip:
        y = np.clip(y, -1.0, 1.0)

    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)

    return y


import re

def chunk_text(text: str, max_chars: int = 200) -> list:
    """
    Split text into chunks without cutting words.
    Tries to split on sentence boundaries first, then falls back to word boundaries.
    """
    text = text.strip()
    if not text:
        return []
    
    if len(text) <= max_chars:
        return [text]
    
    # Sentence-ending punctuation patterns
    sentence_endings = re.compile(r'(?<=[.!?。！？])\s+')
    
    # Split into sentences first
    sentences = sentence_endings.split(text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # If single sentence is too long, split by words
        if len(sentence) > max_chars:
            # Flush current chunk first
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            # Split long sentence by words
            words = sentence.split()
            for word in words:
                if len(current_chunk) + len(word) + 1 <= max_chars:
                    current_chunk = current_chunk + " " + word if current_chunk else word
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = word
        else:
            # Try to add sentence to current chunk
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            if len(test_chunk) <= max_chars:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def _audio_to_tuple(audio):
    """Convert Gradio audio input to (wav, sr) tuple."""
    if audio is None:
        return None

    if isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[0], int):
        sr, wav = audio
        wav = _normalize_audio(wav)
        return wav, int(sr)

    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        sr = int(audio["sampling_rate"])
        wav = _normalize_audio(audio["data"])
        return wav, sr

    return None


# Speaker and language choices for CustomVoice model
SPEAKERS = [
    "Aiden", "Dylan", "Eric", "Ono_anna", "Ryan", "Serena", "Sohee", "Uncle_fu", "Vivian"
]
LANGUAGES = ["Auto", "Italian", "Chinese", "English", "Japanese", "Korean", "French", "German", "Spanish", "Portuguese", "Russian"]


import random

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Force deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def generate_voice_design(text, language, voice_description, seed):
    """Generate speech using Voice Design model (1.7B only)."""
    if not text or not text.strip():
        return None, "Error: Text is required."
    if not voice_description or not voice_description.strip():
        return None, "Error: Voice description is required."

    try:
        # Handle seed - if -1 (auto), generate one
        if seed == -1:
            seed = random.randint(0, 2147483647)
        seed = int(seed)
        set_seed(seed)
        
        tts = get_model("VoiceDesign", "1.7B")
        
        print(f"\n{'='*50}")
        print(f"🎨 Voice Design Generation")
        print(f"{'='*50}")
        print(f"🎲 Seed: {seed}")
        print(f"📝 Text length: {len(text)} chars")
        
        wavs, sr = tts.generate_voice_design(
            text=text.strip(),
            language=language,
            instruct=voice_description.strip(),
            non_streaming_mode=True,
            max_new_tokens=2048,
        )
        
        total_duration = len(wavs[0]) / sr
        print(f"\n{'='*50}")
        print(f"✅ Complete! Duration: {total_duration:.2f}s")
        print(f"{'='*50}\n")
        
        status = f"Generated {total_duration:.1f}s of audio | Seed: {seed}"
        return (sr, wavs[0]), status
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        return None, f"Error: {type(e).__name__}: {e}"


def generate_voice_clone(ref_audio, ref_text, target_text, language, use_xvector_only, model_size, max_chunk_chars, chunk_gap, seed):
    """Generate speech using Base (Voice Clone) model."""
    if not target_text or not target_text.strip():
        return None, "Error: Target text is required."

    audio_tuple = _audio_to_tuple(ref_audio)
    if audio_tuple is None:
        return None, "Error: Reference audio is required."

    if not use_xvector_only and (not ref_text or not ref_text.strip()):
        return None, "Error: Reference text is required when 'Use x-vector only' is not enabled."

    try:
        from tqdm import tqdm
        
        # Handle seed - if -1 (auto), generate one and use it for all chunks
        if seed == -1:
            seed = random.randint(0, 2147483647)
        seed = int(seed)
        
        tts = get_model("Base", model_size)
        chunks = chunk_text(target_text.strip(), max_chars=int(max_chunk_chars))
        
        print(f"\n{'='*50}")
        print(f"🎭 Voice Clone Generation ({model_size})")
        print(f"{'='*50}")
        print(f"🎲 Seed: {seed}")
        print(f"📝 Text length: {len(target_text)} chars → {len(chunks)} chunk(s)")
        print(f"⏱️ Chunk gap: {chunk_gap}s")
        
        all_wavs = []
        sr = None
        for i, chunk in enumerate(tqdm(chunks, desc="Generating chunks", unit="chunk")):
            # Set seed before each chunk to ensure consistency
            set_seed(seed)
            
            print(f"\n🔊 Chunk {i+1}/{len(chunks)} [Seed: {seed}]: \"{chunk[:50]}{'...' if len(chunk) > 50 else ''}\"")
            wavs, sr = tts.generate_voice_clone(
                text=chunk,
                language=language,
                ref_audio=audio_tuple,
                ref_text=ref_text.strip() if ref_text else None,
                x_vector_only_mode=use_xvector_only,
                max_new_tokens=2048,
            )
            all_wavs.append(wavs[0])
            print(f"   ✅ Generated {len(wavs[0])/sr:.2f}s of audio")
        
        # Concatenate all audio chunks with gap (silence) between them
        if len(all_wavs) > 1 and chunk_gap > 0:
            gap_samples = int(sr * chunk_gap)
            silence = np.zeros(gap_samples, dtype=np.float32)
            chunks_with_gaps = []
            for i, wav in enumerate(all_wavs):
                chunks_with_gaps.append(wav)
                if i < len(all_wavs) - 1:  # Don't add gap after last chunk
                    chunks_with_gaps.append(silence)
            final_wav = np.concatenate(chunks_with_gaps)
        else:
            final_wav = np.concatenate(all_wavs) if len(all_wavs) > 1 else all_wavs[0]
        
        total_duration = len(final_wav) / sr
        print(f"\n{'='*50}")
        print(f"✅ Complete! Total duration: {total_duration:.2f}s")
        print(f"{'='*50}\n")
        
        status = f"Generated {len(chunks)} chunk(s), {total_duration:.1f}s total | Seed: {seed}" if len(chunks) > 1 else f"Generated {total_duration:.1f}s of audio | Seed: {seed}"
        return (sr, final_wav), status
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        return None, f"Error: {type(e).__name__}: {e}"


def generate_custom_voice(text, language, speaker, instruct, model_size, seed):
    """Generate speech using CustomVoice model."""
    if not text or not text.strip():
        return None, "Error: Text is required."
    if not speaker:
        return None, "Error: Speaker is required."

    try:
        # Handle seed - if -1 (auto), generate one
        if seed == -1:
            seed = random.randint(0, 2147483647)
        seed = int(seed)
        set_seed(seed)
        
        tts = get_model("CustomVoice", model_size)
        
        print(f"\n{'='*50}")
        print(f"🗣️ Custom Voice Generation ({model_size})")
        print(f"{'='*50}")
        print(f"🎲 Seed: {seed}")
        print(f"👤 Speaker: {speaker}")
        print(f"📝 Text length: {len(text)} chars")
        
        wavs, sr = tts.generate_custom_voice(
            text=text.strip(),
            language=language,
            speaker=speaker.lower().replace(" ", "_"),
            instruct=instruct.strip() if instruct else None,
            non_streaming_mode=True,
            max_new_tokens=2048,
        )
        
        total_duration = len(wavs[0]) / sr
        print(f"\n{'='*50}")
        print(f"✅ Complete! Duration: {total_duration:.2f}s")
        print(f"{'='*50}\n")
        
        status = f"Generated {total_duration:.1f}s of audio | Seed: {seed}"
        return (sr, wavs[0]), status
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        return None, f"Error: {type(e).__name__}: {e}"


# Build Gradio UI
def build_ui():
    theme = gr.themes.Soft(
        font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"],
        primary_hue="indigo",
        secondary_hue="slate",
    )

    css = """
    .gradio-container {
        max-width: 100% !important;
        padding: 0 2rem !important;
    }
    .header-container {
        text-align: center;
        padding: 2rem 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        margin-bottom: 1.5rem;
    }
    .header-container h1 {
        color: white !important;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    .header-container p {
        color: rgba(255,255,255,0.9) !important;
        font-size: 1.1rem !important;
    }
    .feature-badge {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.85rem;
        color: white;
    }
    .tab-content {
        min-height: 500px !important;
    }
    .tabitem {
        min-height: 500px !important;
    }
    """

    with gr.Blocks(title="Qwen3-TTS Demo") as demo:
        gr.HTML(
            """
            <div class="header-container">
                <h1>🎙️ Qwen3-TTS</h1>
                <p>High-Quality Text-to-Speech with Voice Cloning & Design</p>
                <div style="margin-top: 1rem;">
                    <span class="feature-badge">🎨 Voice Design</span>
                    <span class="feature-badge">🎭 Voice Clone</span>
                    <span class="feature-badge">🗣️ Custom Voices</span>
                    <span class="feature-badge">📝 Long Text Chunking</span>
                </div>
            </div>
            """
        )

        with gr.Tabs():
            # Tab 0: Model Management (Collapsible sections)
            with gr.Tab("⚙️ Models"):
                with gr.Accordion("📥 Download Models", open=True):
                    gr.Markdown("*💡 Tip: Models can be downloaded here or will auto-download when you generate in any tab.*")
                    with gr.Row():
                        with gr.Column(scale=1):
                            with gr.Row():
                                download_model_type = gr.Dropdown(
                                    label="Type",
                                    choices=list(AVAILABLE_MODELS.keys()),
                                    value="CustomVoice",
                                    interactive=True,
                                    scale=2,
                                )
                                download_model_size = gr.Dropdown(
                                    label="Size",
                                    choices=["0.6B", "1.7B"],
                                    value="1.7B",
                                    interactive=True,
                                    scale=1,
                                )
                            download_btn = gr.Button("Download", variant="primary", size="sm")
                            download_status = gr.Textbox(label="Status", lines=1, interactive=False)
                        with gr.Column(scale=2):
                            models_status = gr.Markdown(value=get_downloaded_models_status)
                
                download_model_type.change(
                    get_available_sizes,
                    inputs=[download_model_type],
                    outputs=[download_model_size],
                )
                
                download_btn.click(
                    download_model,
                    inputs=[download_model_type, download_model_size],
                    outputs=[download_status, models_status],
                )
                
                with gr.Accordion("🚀 Load Models to GPU", open=False):
                    with gr.Row():
                        with gr.Column(scale=1):
                            with gr.Row():
                                load_model_type = gr.Dropdown(
                                    label="Type",
                                    choices=list(AVAILABLE_MODELS.keys()),
                                    value="CustomVoice",
                                    interactive=True,
                                    scale=2,
                                )
                                load_model_size = gr.Dropdown(
                                    label="Size",
                                    choices=["0.6B", "1.7B"],
                                    value="1.7B",
                                    interactive=True,
                                    scale=1,
                                )
                            load_btn = gr.Button("Load to GPU", variant="primary", size="sm")
                            load_status = gr.Textbox(label="Status", lines=1, interactive=False)
                        with gr.Column(scale=2):
                            load_refresh_btn = gr.Button("🔄 Refresh Status", size="sm")
                            load_loaded_status = gr.Markdown(value=get_loaded_models_status)
                
                load_model_type.change(
                    get_available_sizes,
                    inputs=[load_model_type],
                    outputs=[load_model_size],
                )
                
                load_refresh_btn.click(
                    lambda: get_loaded_models_status(),
                    inputs=[],
                    outputs=[load_loaded_status],
                )
                
                load_btn.click(
                    load_model_manual,
                    inputs=[load_model_type, load_model_size],
                    outputs=[load_status, load_loaded_status],
                )
                
                with gr.Accordion("🗑️ Unload Models", open=False):
                    gr.Markdown("*💡 Tip: Click 'Refresh Status' to see models loaded from other tabs.*")
                    with gr.Row():
                        with gr.Column(scale=1):
                            with gr.Row():
                                unload_model_type = gr.Dropdown(
                                    label="Type",
                                    choices=list(AVAILABLE_MODELS.keys()),
                                    value="CustomVoice",
                                    interactive=True,
                                    scale=2,
                                )
                                unload_model_size = gr.Dropdown(
                                    label="Size",
                                    choices=["0.6B", "1.7B"],
                                    value="1.7B",
                                    interactive=True,
                                    scale=1,
                                )
                            with gr.Row():
                                unload_btn = gr.Button("Unload Selected", variant="secondary", size="sm")
                                unload_all_btn = gr.Button("Unload All", variant="stop", size="sm")
                            unload_status = gr.Textbox(label="Status", lines=1, interactive=False)
                        with gr.Column(scale=2):
                            refresh_btn = gr.Button("🔄 Refresh Status", size="sm")
                            loaded_status = gr.Markdown(value=get_loaded_models_status)
                
                unload_model_type.change(
                    get_available_sizes,
                    inputs=[unload_model_type],
                    outputs=[unload_model_size],
                )
                
                refresh_btn.click(
                    lambda: get_loaded_models_status(),
                    inputs=[],
                    outputs=[loaded_status],
                )
                
                unload_btn.click(
                    unload_model,
                    inputs=[unload_model_type, unload_model_size],
                    outputs=[unload_status, loaded_status],
                )
                
                unload_all_btn.click(
                    unload_all_models,
                    inputs=[],
                    outputs=[unload_status, loaded_status],
                )

            # Tab 1: Voice Design
            with gr.Tab("🎨 Voice Design"):
                gr.Markdown("*ℹ️ Voice Design generates unique voices from descriptions. Max ~2048 tokens (~300-500 chars recommended). No chunking - for longer texts use Voice Clone or Custom Voice.*")
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        design_text = gr.Textbox(
                            label="Text to Synthesize",
                            lines=6,
                            placeholder="Enter the text you want to convert to speech (keep under ~500 chars)...",
                            value="It's in the top drawer... wait, it's empty? No way, that's impossible! I'm sure I put it there!"
                        )
                        design_instruct = gr.Textbox(
                            label="Voice Description",
                            lines=3,
                            placeholder="Describe the voice characteristics you want...",
                            value="Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice."
                        )
                        with gr.Row():
                            design_language = gr.Dropdown(
                                label="Language",
                                choices=LANGUAGES,
                                value="Auto",
                                interactive=True,
                            )
                            design_seed = gr.Number(
                                label="Seed (-1 = Auto)",
                                value=-1,
                                precision=0,
                            )
                        design_btn = gr.Button("🎙️ Generate", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        design_audio_out = gr.Audio(label="Generated Audio", type="numpy")
                        design_status = gr.Textbox(label="Status", lines=2, interactive=False)

                design_btn.click(
                    generate_voice_design,
                    inputs=[design_text, design_language, design_instruct, design_seed],
                    outputs=[design_audio_out, design_status],
                )

            # Tab 2: Voice Clone
            with gr.Tab("🎭 Voice Clone"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        clone_ref_audio = gr.Audio(
                            label="Reference Audio",
                            type="numpy",
                        )
                        with gr.Row():
                            clone_ref_text = gr.Textbox(
                                label="Reference Text",
                                lines=2,
                                placeholder="Transcript of reference audio...",
                                scale=3,
                            )
                            transcribe_btn = gr.Button("🎤 Transcribe", scale=1)
                        clone_xvector = gr.Checkbox(
                            label="X-vector only (no text needed, lower quality)",
                            value=False,
                        )
                        clone_target_text = gr.Textbox(
                            label="Target Text",
                            lines=5,
                            placeholder="Text to synthesize with cloned voice...",
                        )
                        with gr.Row():
                            clone_language = gr.Dropdown(
                                label="Language",
                                choices=LANGUAGES,
                                value="Auto",
                                interactive=True,
                            )
                            clone_model_size = gr.Dropdown(
                                label="Size",
                                choices=MODEL_SIZES,
                                value="1.7B",
                                interactive=True,
                            )
                        with gr.Row():
                            clone_chunk_size = gr.Slider(
                                label="Chunk Size",
                                minimum=50,
                                maximum=500,
                                value=200,
                                step=10,
                            )
                            clone_chunk_gap = gr.Slider(
                                label="Chunk Gap (s)",
                                minimum=0.0,
                                maximum=3.0,
                                value=0.0,
                                step=0.01,
                            )
                        with gr.Row():
                            clone_seed = gr.Number(
                                label="Seed (-1 = Auto)",
                                value=-1,
                                precision=0,
                            )
                        clone_btn = gr.Button("🎙️ Clone & Generate", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        clone_audio_out = gr.Audio(label="Generated Audio", type="numpy")
                        clone_status = gr.Textbox(label="Status", lines=2, interactive=False)

                transcribe_btn.click(
                    transcribe_audio,
                    inputs=[clone_ref_audio],
                    outputs=[clone_ref_text],
                )
                
                clone_btn.click(
                    generate_voice_clone,
                    inputs=[clone_ref_audio, clone_ref_text, clone_target_text, clone_language, clone_xvector, clone_model_size, clone_chunk_size, clone_chunk_gap, clone_seed],
                    outputs=[clone_audio_out, clone_status],
                )

            # Tab 3: Custom Voice TTS
            with gr.Tab("🗣️ Custom Voice"):
                gr.Markdown("*ℹ️ Custom Voice uses predefined speakers. Max ~2048 tokens (~300-500 chars recommended). For longer texts use Voice Clone.*")
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        tts_text = gr.Textbox(
                            label="Text to Synthesize",
                            lines=6,
                            placeholder="Enter the text you want to convert to speech (keep under ~500 chars)...",
                            value="Hello! Welcome to the Text-to-Speech system. This is a demo of our TTS capabilities."
                        )
                        with gr.Row():
                            tts_speaker = gr.Dropdown(
                                label="Speaker",
                                choices=SPEAKERS,
                                value="Ryan",
                                interactive=True,
                            )
                            tts_language = gr.Dropdown(
                                label="Language",
                                choices=LANGUAGES,
                                value="English",
                                interactive=True,
                            )
                        tts_instruct = gr.Textbox(
                            label="Style Instruction (Optional, 1.7B only)",
                            lines=2,
                            placeholder="e.g., Speak in a cheerful and energetic tone",
                        )
                        with gr.Row():
                            tts_model_size = gr.Dropdown(
                                label="Size",
                                choices=MODEL_SIZES,
                                value="1.7B",
                                interactive=True,
                            )
                            tts_seed = gr.Number(
                                label="Seed (-1 = Auto)",
                                value=-1,
                                precision=0,
                            )
                        tts_btn = gr.Button("🎙️ Generate", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        tts_audio_out = gr.Audio(label="Generated Audio", type="numpy")
                        tts_status = gr.Textbox(label="Status", lines=2, interactive=False)

                tts_btn.click(
                    generate_custom_voice,
                    inputs=[tts_text, tts_language, tts_speaker, tts_instruct, tts_model_size, tts_seed],
                    outputs=[tts_audio_out, tts_status],
                )

    return demo, theme, css


if __name__ == "__main__":
    demo, theme, css = build_ui()
    demo.launch(theme=theme, css=css)
