"""Make app/app.py fall back to CPU when torch has no CUDA support.

Upstream app.py loads models with device_map="cuda", which raises
"AssertionError: Torch not compiled with CUDA enabled" on CPU/DirectML/ROCm-less
installs. Idempotent: safe to run after every install and update.
"""
from pathlib import Path

APP = Path(__file__).parent / "app" / "app.py"
OLD = 'device_map="cuda",'
NEW = 'device_map="cuda" if torch.cuda.is_available() else "cpu",'

src = APP.read_text(encoding="utf-8")
if OLD in src:
    APP.write_text(src.replace(OLD, NEW), encoding="utf-8")
    print("patch_app: applied CUDA fallback")
else:
    print("patch_app: nothing to patch")
