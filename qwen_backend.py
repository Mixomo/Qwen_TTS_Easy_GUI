"""Qwen3-TTS engine and workflow services for Qwen TTS Easy GUI.

The UI deliberately stays close to the Fish/XTTS Easy GUI workflow, while
this module owns Qwen-specific model routing, voice prompts, model downloads,
dataset JSONL preparation, the Qwen-compatible full-SFT path and
optional PEFT adapter loading.
All heavyweight ML imports are lazy so the application can still open and
explain a missing runtime instead of failing at import time.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from functools import wraps
from pathlib import Path

from qwen_easy.console import log

try:
    import winsound
except ImportError:  # pragma: no cover - keeps structural tests portable.
    winsound = None

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
BASE_MODELS = ROOT / "base_models"
FAST_MERGED_MODELS = MODELS / "fast_merged"
OUTPUTS = ROOT / "outputs"
SAMPLES = ROOT / "samples"
DATASETS = ROOT / "datasets"
PROJECTS = ROOT / "projects"
TRAINING = ROOT / "training"
INSTRUCTIONS = ROOT / "instructions"
RUNTIME = ROOT / ".runtime"
FINETUNING = ROOT / "modules" / "qwen3_tts" / "finetuning"
CHIME_PATH = ROOT / "assets" / "inference_training_done.wav"
for _folder in (MODELS, BASE_MODELS, FAST_MERGED_MODELS, OUTPUTS, SAMPLES, DATASETS, PROJECTS, TRAINING, INSTRUCTIONS, RUNTIME):
    _folder.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(RUNTIME / "cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

NONE = "None"
SAMPLE_RATE = 24000
LANGUAGES = ["Auto", "Chinese", "English", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"]
LANGUAGE_CHOICES = [(value, value) for value in LANGUAGES]
SPEAKERS = ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee"]
MODES = ["CustomVoice", "VoiceDesign", "Base / Voice Clone", "Trained Custom Voice"]
CHUNK_CHOICES = ["None", "Periods", "Lines", "Paragraphs"]
ATTENTION_CHOICES = ["Auto (FlashAttention 2 → SDPA → eager)", "FlashAttention 2", "SDPA", "eager"]
TRAINING_ATTENTION_CHOICES = ["Auto (SDPA → eager · stable training)", "FlashAttention 2 (opt-in)", "SDPA", "eager"]
TRAINING_LR_SCHEDULE_CHOICES = [
    ("Warmup + cosine decay (recommended)", "warmup_cosine"),
    ("Constant LR (legacy / diagnostic)", "constant"),
]
MODEL_EXPORT_MODE_CHOICES = ["Non-ICL-variant", "ICL-variant", "Both"]
# PEFT does not publish a full model-export variant.  The adapter remains
# Base-compatible and therefore retains Base ICL when loaded with its Base.
PEFT_ADAPTER_EXPORT_MODE = "PEFT adapter (Base ICL compatible)"
# New full-SFT runs publish one canonical hybrid Base checkpoint.  The other
# values remain accepted by the compatibility layer so older project files and
# hand-written worker commands can still be read safely.
DEFAULT_MODEL_EXPORT_MODE = "ICL-variant"
INFERENCE_ENGINE_CHOICES = ["Standard Qwen", "Faster Qwen CUDA Graphs"]
DEFAULT_INFERENCE_ENGINE = "Standard Qwen"
FRESH_RESUME = "Fresh / None"
ICL_TRAILING_SILENCE_SECONDS = 0.5
PACKAGED_SPEAKER_PROMPT_FILE = "trained_speaker_embedding.safetensors"
PACKAGED_SPEAKER_PROMPT_METADATA = "trained_speaker_prompt.json"
PACKAGED_ICL_PROMPT_FILE = "trained_icl_prompt.safetensors"
PACKAGED_ICL_PROMPT_METADATA = "trained_icl_prompt.json"
PACKAGED_SPEAKER_CHOICE = "__PACKAGED_TRAINED_SPEAKER__"
DEFAULT_TOKENIZER_MODEL = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
SAMPLE_AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}
WHISPER_MODELS = {
    "large-v3 (~10 GB VRAM)": "large-v3",
    "large-v2 (~10 GB VRAM)": "large-v2",
    "large-v3-turbo (~6 GB VRAM)": "large-v3-turbo",
    "medium (~5 GB VRAM)": "medium",
    "small (~2 GB VRAM)": "small",
    "base (~1 GB VRAM)": "base",
    "tiny (~1 GB VRAM)": "tiny",
}
DEFAULT_WHISPER_MODEL = "large-v3 (~10 GB VRAM)"
TRAINING_HARDWARE_PRESETS = [
    "12 GB VRAM (0.6B only)",
    "16 GB VRAM (0.6B recommended)",
    "24 GB VRAM (1.7B minimum)",
    "32 GB+ VRAM (1.7B headroom)",
]
DEFAULT_INSTRUCTIONS = {
    "None": "",
    "Warm narration": "Warm, natural narration with relaxed pacing and clear diction.",
    "Friendly conversational": "Friendly, conversational and approachable, with a light smile in the voice.",
    "Calm documentary": "Calm, measured documentary delivery with steady pacing and precise articulation.",
    "Bright and energetic": "Bright, energetic and expressive, with lively rhythm and confident emphasis.",
    "Soft and intimate": "Soft, intimate delivery with gentle energy and close, natural phrasing.",
}

# The official Qwen repository currently publishes this five-model 12 Hz
# family. The parenthesized figures are approximate loading budgets for the
# Windows BF16 runtime, not vendor guarantees; actual peak VRAM depends on
# text length, attention backend and generation settings.
MODEL_CATALOG = {
    "12Hz · 1.7B · VoiceDesign (≈6 GB VRAM)": {"id": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "mode": "VoiceDesign", "size": "1.7B", "hz": 12, "vram": 6, "supports_icl": False},
    "12Hz · 1.7B · CustomVoice (≈6 GB VRAM)": {"id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "mode": "CustomVoice", "size": "1.7B", "hz": 12, "vram": 6, "supports_icl": False},
    "12Hz · 1.7B · Base / Voice Clone (≈6 GB VRAM)": {"id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "mode": "Base / Voice Clone", "size": "1.7B", "hz": 12, "vram": 6, "supports_icl": True},
    "12Hz · 0.6B · CustomVoice (≈4 GB VRAM)": {"id": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "mode": "CustomVoice", "size": "0.6B", "hz": 12, "vram": 4, "supports_icl": False},
    "12Hz · 0.6B · Base / Voice Clone (≈4 GB VRAM)": {"id": "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "mode": "Base / Voice Clone", "size": "0.6B", "hz": 12, "vram": 4, "supports_icl": True},
}

_MODEL = None
_MODEL_KEY = None
_MODEL_ADAPTER = None
_FAST_MODEL = None
_FAST_MODEL_KEY = None
_MODEL_LOCK = threading.RLock()
_PROMPT_CACHE = {}
_ASR = None
_ASR_KEY = None
_CANCEL_AUX = threading.Event()
_INFERENCE_CANCEL = threading.Event()
_INFERENCE_LOCK = threading.RLock()
_INFERENCE_ACTIVE_JOBS = 0
_TRAIN_PROC = None
_TRAIN_THREAD = None
_TRAIN_LOCK = threading.RLock()
_TRAIN_STATE = {
    "running": False, "starting": False, "status": "Idle", "project": "",
    "started": None, "finished_elapsed": None, "returncode": None, "log": "", "output_dir": "",
    "epoch": 0, "total_epochs": 0, "step": 0, "loss": None,
    "eval_epoch": 0, "eval_step": 0, "eval_seconds": None, "eval_elapsed": None, "nonfinite": False,
    "completion_chimed": False,
}


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip()).strip("._-") or "item"


def normalize_model_export_mode(value=None, legacy_publish_icl=None):
    """Normalize the export selector and migrate the former boolean setting."""
    if isinstance(value, bool):
        legacy_publish_icl = value
        value = None
    if value is None or str(value).strip() == "":
        if legacy_publish_icl is not None:
            return "Both" if bool(legacy_publish_icl) else "Non-ICL-variant"
        return DEFAULT_MODEL_EXPORT_MODE
    candidate = str(value).strip()
    aliases = {
        "Non-ICL variant": "Non-ICL-variant",
        "non-ICL-variant": "Non-ICL-variant",
        "non-icl-variant": "Non-ICL-variant",
        "ICL variant": "ICL-variant",
        "icl-variant": "ICL-variant",
    }
    candidate = aliases.get(candidate, candidate)
    if candidate == PEFT_ADAPTER_EXPORT_MODE:
        return candidate
    return candidate if candidate in MODEL_EXPORT_MODE_CHOICES else DEFAULT_MODEL_EXPORT_MODE


def play_done_chime() -> None:
    """Play the same non-blocking WAV pattern used by Fish and XTTS."""
    chime_path = str(CHIME_PATH)
    if os.path.exists(chime_path):
        try:
            if winsound is not None:
                # Keep this call deliberately identical to the working sister
                # GUIs: filename playback plus SND_ASYNC, without extra flags
                # that can make Windows silently reject a WAV in some sessions.
                winsound.PlaySound(chime_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                log(f"[UI] Completion chime triggered: {chime_path}")
        except Exception as exc:
            log(f"[UI] Completion chime unavailable: {exc}", level="WARN")
    else:
        if winsound is not None:
            try:
                winsound.MessageBeep()
                log("[UI] Completion chime fallback beep triggered.")
            except Exception as exc:
                log(f"[UI] Completion fallback beep unavailable: {exc}", level="WARN")


def whisper_model_id(value):
    """Resolve a human-readable Faster-Whisper label to its model id."""
    raw = str(value or "small")
    if raw in WHISPER_MODELS:
        return WHISPER_MODELS[raw]
    # Accept values produced by older builds that incorrectly sanitized the
    # visible VRAM-qualified label before reaching Faster-Whisper.
    for label, model_id in WHISPER_MODELS.items():
        if _safe(label) == raw:
            return model_id
    return raw


def _estimated_vram(size):
    normalized = str(size).lower().replace("b", ".")
    return 4 if "0.6" in normalized else 6 if "1.7" in normalized else None


def _model_size_from_text(*values):
    """Extract the published Qwen family size from labels and metadata."""
    for value in values:
        match = re.search(r"(?<![0-9])(?:0\.6|1\.7)\s*[Bb](?![A-Za-z0-9])", str(value or ""))
        if match:
            return match.group(0).replace(" ", "").upper()
        # Qwen's raw config currently uses ``1b7`` for the 1.7B family.
        compact = str(value or "").lower().replace(" ", "")
        if "1b7" in compact:
            return "1.7B"
        if "0b6" in compact or "0.6b" in compact:
            return "0.6B"
    return ""


def _instruction_path():
    return INSTRUCTIONS / "library.json"


def _read_instruction_library():
    values = dict(DEFAULT_INSTRUCTIONS)
    path = _instruction_path()
    if path.is_file():
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                values.update({str(k): str(v) for k, v in saved.items() if str(k).strip()})
        except Exception as exc:
            log(f"[UI] Instruction library could not be read: {exc}", level="WARN")
    return values


def list_instruction_choices():
    return [(name, name) for name in sorted(_read_instruction_library(), key=str.casefold)]


def load_instruction(name):
    return _read_instruction_library().get(str(name or "None"), "")


def save_instruction(name, text):
    safe_name = str(name or "").strip()
    value = str(text or "").strip()
    if not safe_name or safe_name == "None":
        return "Give the instruction a name first."
    if not value:
        return "Enter instruction text first."
    values = _read_instruction_library()
    values[safe_name] = value
    _instruction_path().write_text(json.dumps(values, indent=2, ensure_ascii=False), encoding="utf-8")
    return f"Saved instruction '{safe_name}'."


def delete_instruction(name):
    safe_name = str(name or "").strip()
    if not safe_name or safe_name == "None":
        return "Select a saved custom instruction first."
    values = _read_instruction_library()
    if safe_name not in values or safe_name in DEFAULT_INSTRUCTIONS:
        return "Built-in instructions cannot be deleted."
    values.pop(safe_name, None)
    _instruction_path().write_text(json.dumps(values, indent=2, ensure_ascii=False), encoding="utf-8")
    return f"Deleted instruction '{safe_name}'."


def normalize_seed(value):
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return min(value, (1 << 31) - 1)


def random_seed_value():
    return random.randint(1, (1 << 31) - 1)


def apply_generation_seed(value):
    seed = normalize_seed(value)
    if seed is None:
        return None
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    return seed


def begin_aux_job() -> None:
    _CANCEL_AUX.clear()


def stop_aux_job() -> str:
    _CANCEL_AUX.set()
    return "Stop requested. The current audio/dataset operation will stop at the next safe boundary."


def stop_inference() -> str:
    _INFERENCE_CANCEL.set()
    return "Stop requested. Qwen generation cannot be interrupted inside every kernel, but the current workflow will stop between chunks."


def inference_status_snapshot():
    with _INFERENCE_LOCK:
        running = _INFERENCE_ACTIVE_JOBS > 0
    return {"running": running, "stop_requested": bool(_INFERENCE_CANCEL.is_set())}


def _inference_job(function):
    """Track nested single/dialogue generations as one cancellable job."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        global _INFERENCE_ACTIVE_JOBS
        with _INFERENCE_LOCK:
            if _INFERENCE_ACTIVE_JOBS == 0:
                _INFERENCE_CANCEL.clear()
            _INFERENCE_ACTIVE_JOBS += 1
        try:
            result = function(*args, **kwargs)
        finally:
            with _INFERENCE_LOCK:
                _INFERENCE_ACTIVE_JOBS = max(0, _INFERENCE_ACTIVE_JOBS - 1)
                if _INFERENCE_ACTIVE_JOBS == 0:
                    _INFERENCE_CANCEL.clear()
        return result

    return wrapped


def _safe_audio_value(value):
    if not value:
        return []
    values = value if isinstance(value, (list, tuple)) else [value]
    result = []
    for item in values:
        if isinstance(item, dict):
            item = item.get("path") or item.get("name")
        else:
            item = getattr(item, "path", item)
        if item and Path(str(item)).is_file():
            result.append(Path(str(item)))
    return result


def _audio_read(path: Path, target_sr: int = SAMPLE_RATE):
    import numpy as np
    try:
        import soundfile as sf
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception:
        import librosa
        data, sr = librosa.load(str(path), sr=None, mono=True)
    if getattr(data, "ndim", 1) > 1:
        data = np.mean(data, axis=-1)
    data = np.asarray(data, dtype=np.float32)
    if int(sr) != target_sr:
        # Keep the normal WAV path independent from librosa/numba. This is
        # both faster to import on Windows and avoids the NumPy/Numba version
        # coupling seen in several audio environments.
        from scipy.signal import resample_poly
        import math
        divisor = math.gcd(int(sr), int(target_sr))
        data = resample_poly(data, int(target_sr) // divisor, int(sr) // divisor).astype(np.float32)
        sr = target_sr
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak > 1.0:
        data = data / peak
    return data, int(sr)


def _audio_write(path: Path, data, sr: int = SAMPLE_RATE):
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sr, subtype="PCM_16")


def _sidecar_for(audio: Path) -> Path | None:
    for suffix in (".txt", ".lab", ".transcript"):
        sidecar = audio.with_suffix(suffix)
        if sidecar.is_file() and sidecar.read_text(encoding="utf-8", errors="ignore").strip():
            return sidecar
    return None


def _sample_meta(name: str) -> Path:
    return SAMPLES / _safe(name) / "sample.json"


def _flat_sample_audio(name: str) -> Path | None:
    """Find legacy Voice Clone Studio samples stored directly in ``samples``."""
    wanted = str(name or "").casefold()
    if not wanted or not SAMPLES.is_dir():
        return None
    for item in SAMPLES.iterdir():
        if item.is_file() and item.suffix.casefold() in SAMPLE_AUDIO_EXTENSIONS and item.stem.casefold() == wanted:
            return item
    return None


def _flat_sample_sidecar(audio: Path) -> tuple[str, str]:
    """Read the legacy JSON/TXT transcript and language beside one audio file."""
    for suffix in (".json", ".txt", ".lab", ".transcript"):
        sidecar = audio.with_suffix(suffix)
        if not sidecar.is_file():
            continue
        try:
            if sidecar.suffix.casefold() == ".json":
                payload = json.loads(sidecar.read_text(encoding="utf-8", errors="ignore"))
                if isinstance(payload, dict):
                    text = str(
                        payload.get("Text")
                        or payload.get("text")
                        or payload.get("transcript")
                        or payload.get("ref_text")
                        or ""
                    ).strip()
                    language = str(payload.get("Language") or payload.get("language") or "Auto").strip() or "Auto"
                    return text, language
            else:
                text = sidecar.read_text(encoding="utf-8", errors="ignore").strip()
                if text:
                    return text, "Auto"
        except Exception:
            continue
    return "", "Auto"


def _legacy_prompt_path(name: str, model_size: str) -> Path | None:
    """Locate either Voice Clone Studio's ``.prompt`` or equivalent ``.pt``."""
    wanted_stem = f"{name}_{model_size}".casefold()
    if not SAMPLES.is_dir():
        return None
    for item in SAMPLES.iterdir():
        if item.is_file() and item.suffix.casefold() in {".prompt", ".pt"} and item.stem.casefold() == wanted_stem:
            return item
    return None


def _sample_is_multiple(meta: dict) -> bool:
    """Return the persisted reference kind for a Voice Library package.

    ``multiple`` is the current field.  The other markers intentionally make
    the discovery code forward/backward compatible with packages created by
    older builds or by external tools, where the mode may have been stored as
    a human-readable value or inferred from the number of files.
    """
    if not isinstance(meta, dict):
        return False
    if bool(meta.get("multiple")):
        return True
    marker = str(meta.get("sample_type") or meta.get("reference_mode") or "").strip().casefold()
    if marker in {
        "multiple reference mode",
        "multiple audio references",
        "multiple_reference",
        "multiple_reference_package",
        "multiple-reference-package",
    }:
        return True
    files = meta.get("files")
    return isinstance(files, list) and len(files) > 1


def _sample_mode_is_multiple(reference_mode) -> bool:
    return str(reference_mode or "").strip().casefold() in {
        "multiple reference mode",
        "multiple audio references",
    }


def get_sample_entries(reference_mode=None):
    """Discover Voice Library entries with their conditioning kind.

    Managed package folders carry explicit metadata.  Flat ``name.wav``
    samples remain supported as single-reference compatibility entries, but
    are never offered by the multiple-reference filter.
    """
    entries = []
    if not SAMPLES.is_dir():
        return entries

    managed_names = set()
    for item in SAMPLES.iterdir():
        metadata_path = item / "sample.json"
        if not item.is_dir() or not metadata_path.is_file():
            continue
        try:
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        multiple = _sample_is_multiple(meta)
        managed_names.add(item.name.casefold())
        if _sample_mode_is_multiple(reference_mode) and not multiple:
            continue
        if reference_mode and not _sample_mode_is_multiple(reference_mode) and multiple:
            continue
        entries.append({
            "name": item.name,
            "multiple": multiple,
            "sample_type": "multiple_reference_package" if multiple else "single_reference_sample",
            "source": "managed",
        })

    # Keep compatibility with the flat sample layout used by older Qwen
    # interfaces and Voice Clone Studio: ``name.wav`` + ``name.json`` and an
    # optional ``name_1.7B.prompt``/``name_1.7B.pt`` cache.  There is no
    # reliable multiple-reference package in that layout, so it is excluded
    # whenever the Multiple reference mode filter is active.
    if not _sample_mode_is_multiple(reference_mode):
        for item in SAMPLES.iterdir():
            if not item.is_file() or item.suffix.casefold() not in SAMPLE_AUDIO_EXTENSIONS:
                continue
            if item.stem.casefold() in managed_names:
                continue
            entries.append({
                "name": item.stem,
                "multiple": False,
                "sample_type": "single_reference_sample",
                "source": "legacy_flat",
            })
    return sorted(entries, key=lambda entry: str(entry["name"]).casefold())


def get_sample_choices(reference_mode=None):
    return [entry["name"] for entry in get_sample_entries(reference_mode)]


def get_sample_dropdown_choices(reference_mode=None):
    """Return stable dropdown values with a visible reference-kind marker."""
    choices = []
    for entry in get_sample_entries(reference_mode):
        if entry["multiple"]:
            label = f"📦 {entry['name']} · Multiple references"
        else:
            label = f"🎙️ {entry['name']} · Single reference"
        choices.append((label, entry["name"]))
    return choices


def save_sample(audio, name, transcript="", language="Auto", reference_mode="Single reference mode"):
    paths = _safe_audio_value(audio)
    if not paths:
        return "Choose at least one audio file before saving.", None
    safe = _safe(name)
    target = SAMPLES / safe
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True, exist_ok=True)
    normalized = []
    transcripts = []
    for index, path in enumerate(paths if reference_mode == "Multiple reference mode" else paths[:1], start=1):
        out = target / f"reference_{index:02d}.wav"
        data, sr = _audio_read(path)
        _audio_write(out, data, sr)
        normalized.append(str(out))
        sidecar = _sidecar_for(path)
        transcripts.append(sidecar.read_text(encoding="utf-8", errors="ignore").strip() if sidecar else str(transcript or "").strip())
    multiple = reference_mode == "Multiple reference mode"
    meta = {
        "name": safe,
        "multiple": multiple,
        "sample_type": "multiple_reference_package" if multiple else "single_reference_sample",
        "reference_mode": reference_mode,
        "transcript": str(transcript or "").strip(),
        "transcripts": transcripts,
        "language": language or "Auto",
        "files": normalized,
        "created_at": datetime.now().isoformat(),
    }
    (target / "sample.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"[UI] Saved Voice Library entry '{safe}' ({len(normalized)} reference(s)).")
    return f"Saved '{safe}' with {len(normalized)} reference audio file(s).", safe


def load_sample(name):
    if not name or name == NONE:
        return None, "", "Auto", "No saved voice selected."
    path = _sample_meta(name)
    if path.is_file():
        meta = json.loads(path.read_text(encoding="utf-8"))
        files = [str(Path(item)) for item in meta.get("files", []) if Path(item).is_file()]
        if not files:
            files = [str(x) for x in sorted(path.parent.glob("*.wav"))]
        value = files if _sample_is_multiple(meta) else (files[0] if files else None)
        return value, meta.get("transcript", ""), meta.get("language", "Auto"), f"Loaded '{name}' ({len(files)} reference(s))."
    audio = _flat_sample_audio(name)
    if audio is None:
        return None, "", "Auto", f"Voice '{name}' was not found."
    transcript, language = _flat_sample_sidecar(audio)
    return str(audio), transcript, language, f"Loaded legacy Voice Library sample '{name}' (1 reference)."


def delete_sample(name):
    if not name or name == NONE:
        return "Select a saved voice first."
    folder = SAMPLES / _safe(name)
    if not folder.exists():
        return f"Voice '{name}' was not found."
    shutil.rmtree(folder, ignore_errors=True)
    _PROMPT_CACHE.clear()
    return f"Deleted Voice Library entry '{name}'."


def list_local_model_paths():
    candidates = []
    for root in (MODELS, BASE_MODELS, TRAINING):
        if not root.exists():
            continue
        for config in root.rglob("config.json"):
            folder = config.parent
            if folder.name.lower() in {"speech_tokenizer", "tokenizer"}:
                continue
            # Training runs are resume/checkpoint workspaces, not published
            # inference models. Only ready/ready_icl and model-library roots
            # should appear in the model selector; this also prevents the
            # source icl_variant from duplicating the published ready_icl.
            if any(part.casefold() == "runs" for part in folder.parts):
                continue
            if (folder / "model.safetensors").is_file() or any(folder.glob("*.bin")) or (folder / "model.safetensors.index.json").is_file():
                candidates.append(folder)
    return sorted(set(candidates), key=lambda x: str(x).casefold())


def list_ready_models():
    choices = [(label, label) for label in MODEL_CATALOG]
    catalog_local_names = {
        _safe(info["id"].replace("/", "--")).casefold()
        for info in MODEL_CATALOG.values()
    }
    for folder in list_local_model_paths():
        if any(Path(label).resolve() == folder.resolve() for _label, label in choices if str(label).startswith("local::")):
            continue
        # A downloaded official checkpoint is already represented by its
        # catalog entry. The catalog loader resolves that entry to MODELS or
        # BASE_MODELS, so exposing the same directory again as "Local" only
        # creates a confusing duplicate selection.
        if folder.parent in {MODELS, BASE_MODELS} and folder.name.casefold() in catalog_local_names:
            continue
        local_size = ""
        local_kind = ""
        metadata = {}
        try:
            local_config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
            local_size = str(local_config.get("tts_model_size", ""))
            local_kind = str(local_config.get("tts_model_type", ""))
        except Exception:
            pass
        metadata_path = folder / "qwen_easy_training.json"
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                metadata = {}
        model_size = _model_size_from_text(
            metadata.get("model_size"),
            metadata.get("base_model_choice"),
            metadata.get("base_model"),
            metadata.get("source_base_model"),
            local_size,
            folder,
        )
        vram = _estimated_vram(model_size or local_size)
        size_label = f" · {model_size}" if model_size else ""
        label = "Local · " + folder.name + size_label + (f" (≈{vram} GB VRAM)" if vram else "")
        if folder.name == "ready_icl" or metadata.get("variant") == "base_icl":
            project_name = str(metadata.get("project") or folder.parent.name)
            label = "Unofficial Hybrid Trained · " + project_name + size_label + (f" (≈{vram} GB VRAM)" if vram else "")
        elif folder.name == "ready" or "ready" in folder.parts:
            label = "Trained Custom Voice · " + folder.parent.name + size_label + (f" (≈{vram} GB VRAM)" if vram else "")
        choices.append((label, f"local::{folder}"))
    return choices


def list_trained_checkpoint_choices(model_choice):
    """Return the published model and compatible intermediate checkpoints.

    Training checkpoints stay inside the project run directory and are not
    added to the global model catalog. They are exposed here only when the
    selected model is a published trained artifact, keeping the main model
    dropdown compact while allowing checkpoint-by-checkpoint inference.
    """
    if not str(model_choice or "").startswith("local::"):
        return []
    try:
        selected_path = Path(str(model_choice)[7:]).resolve()
        info = _catalog_info(model_choice)
    except Exception:
        return []
    is_trained = info.get("mode") == "Trained Custom Voice" or info.get("variant") == "base_icl"
    if not is_trained:
        return []

    metadata = {}
    metadata_path = selected_path / "qwen_easy_training.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}
    project = str(metadata.get("project") or "").strip()
    if not project and selected_path.name in {"ready", "ready_icl"}:
        project = selected_path.parent.name
    if not project:
        return [("Published checkpoint", str(model_choice))]

    runs_root = TRAINING / _safe(project) / "runs"
    choices = [("Published checkpoint", str(model_choice))]
    if not runs_root.is_dir():
        return choices

    root_variant = str(info.get("variant") or "")
    candidates = []
    for run_dir in sorted((p for p in runs_root.iterdir() if p.is_dir()), key=lambda p: p.name.casefold(), reverse=True):
        for checkpoint in sorted(run_dir.glob("checkpoint-*"), key=lambda p: p.name.casefold(), reverse=True):
            config_path = checkpoint / "config.json"
            complete = config_path.is_file() and (
                (checkpoint / "model.safetensors").is_file()
                or bool(list(checkpoint.glob("*.bin")))
                or (checkpoint / "model.safetensors.index.json").is_file()
            )
            if not complete:
                continue
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                config = {}
            checkpoint_metadata = {}
            checkpoint_metadata_path = checkpoint / "qwen_easy_training.json"
            if checkpoint_metadata_path.is_file():
                try:
                    checkpoint_metadata = json.loads(checkpoint_metadata_path.read_text(encoding="utf-8"))
                except Exception:
                    checkpoint_metadata = {}
            checkpoint_is_icl = (
                config.get("tts_model_type") == "base"
                or checkpoint_metadata.get("variant") == "base_icl"
                or checkpoint_metadata.get("hybrid_checkpoint") is True
            )
            if root_variant == "base_icl" and not checkpoint_is_icl:
                continue
            if root_variant != "base_icl" and checkpoint_is_icl:
                continue
            epoch_match = re.search(r"(\d+)$", checkpoint.name)
            epoch_label = f"Epoch {int(epoch_match.group(1))}" if epoch_match else checkpoint.name
            candidates.append((f"{epoch_label} · {run_dir.name}", f"local::{checkpoint}"))
    choices.extend(candidates)
    return choices


def _catalog_info(choice):
    if choice in MODEL_CATALOG:
        return dict(MODEL_CATALOG[choice], label=choice, path=None)
    if str(choice).startswith("local::"):
        path = Path(str(choice)[7:])
        mode = "Trained Custom Voice"
        size = "local"
        hz = 12
        kind = ""
        metadata = {}
        try:
            cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
            kind = str(cfg.get("tts_model_type", ""))
            if kind == "voice_design":
                mode = "VoiceDesign"
            elif kind == "base":
                mode = "Base / Voice Clone"
            elif kind == "custom_voice":
                mode = "Trained Custom Voice"
            size = str(cfg.get("tts_model_size", size))
            tokenizer_type = str(cfg.get("tokenizer_type", ""))
            if "25hz" in tokenizer_type.lower():
                hz = 25
        except Exception:
            pass
        metadata_path = path / "qwen_easy_training.json"
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                metadata = {}
        model_size = _model_size_from_text(
            metadata.get("model_size"),
            metadata.get("base_model_choice"),
            metadata.get("base_model"),
            metadata.get("source_base_model"),
            size,
            path,
        )
        if model_size:
            size = model_size
        supports_icl = bool(metadata.get("supports_icl", kind == "base"))
        return {
            "id": str(path),
            "mode": mode,
            "size": size,
            "hz": hz,
            "vram": _estimated_vram(size),
            "label": str(choice),
            "path": path,
            "supports_icl": supports_icl,
            "variant": str(metadata.get("variant", "")),
            "project": str(metadata.get("project", "")),
            "speaker_name": str(metadata.get("speaker_name", "")),
        }
    raise ValueError(f"Unknown Qwen model selection: {choice}")


def _model_path_for(choice):
    info = _catalog_info(choice)
    if info.get("path"):
        return Path(info["path"])
    model_id = info["id"]
    safe_id = _safe(model_id.replace("/", "--"))
    for base in (MODELS, BASE_MODELS):
        for candidate in (base / safe_id, base / model_id.split("/")[-1]):
            if (candidate / "config.json").is_file():
                return candidate
    return model_id


def training_resume_choices(project, training_mode=None):
    """Return selectable, compatible checkpoints for a Training project."""
    choices = [(FRESH_RESUME, FRESH_RESUME)]
    if not project or str(project) in {NONE, FRESH_RESUME}:
        return choices
    runs_root = TRAINING / _safe(project) / "runs"
    if not runs_root.is_dir():
        return choices
    want_lora = str(training_mode or "").startswith("PEFT")
    want_full = ("Full SFT" in str(training_mode or "") or str(training_mode or "").startswith("Official")) or not training_mode
    candidates = []
    for run in sorted((p for p in runs_root.iterdir() if p.is_dir()), key=lambda p: p.name.casefold(), reverse=True):
        for checkpoint in sorted(run.glob("checkpoint-*"), key=lambda p: p.name.casefold(), reverse=True):
            is_lora = (checkpoint / "adapter_config.json").is_file()
            is_full = (checkpoint / "config.json").is_file() and bool(list(checkpoint.glob("*.safetensors")) or list(checkpoint.glob("*.bin")) or (checkpoint / "model.safetensors.index.json").is_file())
            if want_lora and not is_lora:
                continue
            if want_full and not is_lora and not is_full:
                continue
            if not want_lora and not want_full and not (is_lora or is_full):
                continue
            kind = "LoRA adapter" if is_lora else "Full SFT checkpoint"
            label = f"{run.name} · {checkpoint.name} · {kind}"
            candidates.append((label, str(checkpoint.resolve())))
    return choices + candidates


def _reset_training_project(project):
    """Remove prior model/run artifacts while preserving prepared datasets.

    ``Fresh / None`` is an explicit destructive reset for one Training
    Project, matching the behavior users expect from the sister GUIs. The
    project metadata under ``projects/`` and all dataset files remain intact;
    only the generated training output under ``training/<project>/`` is
    recreated.
    """
    project_dir = (TRAINING / _safe(project)).resolve()
    training_root = TRAINING.resolve()
    if project_dir == training_root or training_root not in project_dir.parents:
        raise ValueError(f"Unsafe training project path: {project_dir}")
    if project_dir.exists():
        shutil.rmtree(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    log(f"[TRAIN] Fresh / None selected; reset training artifacts for '{project}'.")
    return project_dir


def download_model(choice, progress=None):
    try:
        info = _catalog_info(choice)
        if info.get("path"):
            return f"Already local: {info['path']}"
        from huggingface_hub import snapshot_download
        target = MODELS / _safe(info["id"].replace("/", "--"))
        target.mkdir(parents=True, exist_ok=True)
        log(f"[UI] Downloading {info['id']} into {target}.")
        snapshot_download(repo_id=info["id"], local_dir=str(target), local_dir_use_symlinks=False)
        play_done_chime()
        return f"Downloaded {info['id']} → {target}"
    except Exception as exc:
        log(f"Model download failed: {exc}", level="ERROR")
        return f"Model download failed: {exc}"


def _lora_base_size(adapter_path: Path) -> str:
    """Read the base-family size recorded by a LoRA training run."""
    values = []
    config_path = adapter_path / "adapter_config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            values.append(config.get("base_model_name_or_path", ""))
        except Exception:
            pass
    for parent in (adapter_path, *adapter_path.parents):
        metadata_path = parent / "qwen_easy_lora.json"
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                values.extend((metadata.get("base_model"), metadata.get("base_model_choice"), metadata.get("model_size")))
            except Exception:
                pass
    return _model_size_from_text(*values)


def list_lora_choices(base_choice=None):
    expected_size = ""
    if base_choice:
        try:
            info = _catalog_info(base_choice)
            expected_size = _model_size_from_text(info.get("size"), info.get("id"), info.get("path"))
        except Exception:
            expected_size = ""

    def display_name(adapter_path: Path) -> str:
        """Show the trained project instead of the generic directory name."""
        metadata_project = ""
        for parent in (adapter_path, *adapter_path.parents):
            metadata_path = parent / "qwen_easy_lora.json"
            if metadata_path.is_file():
                try:
                    metadata_project = str(json.loads(metadata_path.read_text(encoding="utf-8")).get("project", "")).strip()
                except Exception:
                    metadata_project = ""
                if metadata_project:
                    break
        try:
            relative = adapter_path.resolve().relative_to(TRAINING.resolve())
            parts = relative.parts
            project = metadata_project or (parts[0] if parts else adapter_path.parent.name)
            details = []
            if len(parts) >= 3 and parts[1].casefold() == "runs":
                details.append(parts[2])
                if len(parts) >= 4 and parts[3].casefold() not in {"adapter", "output"}:
                    details.append(parts[3])
            suffix = " · " + " · ".join(details) if details else ""
            return f"LoRA · {project}{suffix}"
        except Exception:
            return f"LoRA · {metadata_project or adapter_path.parent.name}"

    choices = [(NONE, NONE)]
    roots = [ROOT / "loras", TRAINING]
    for root in roots:
        if not root.exists():
            continue
        for cfg in root.rglob("adapter_config.json"):
            adapter_size = _lora_base_size(cfg.parent)
            if expected_size and adapter_size and adapter_size != expected_size:
                continue
            choices.append((display_name(cfg.parent), str(cfg.parent)))
    seen = set()
    return [item for item in choices if not (item[1] in seen or seen.add(item[1]))]


def _attention_chain(selection):
    if selection == "FlashAttention 2":
        return ["flash_attention_2"]
    if selection == "SDPA":
        return ["sdpa"]
    if selection == "eager":
        return ["eager"]
    return ["flash_attention_2", "sdpa", "eager"]


def _normalize_qwen_speaker_registry(wrapper):
    """Canonicalize speaker IDs for Qwen's case-sensitive low-level lookup.

    Qwen's generation path normalizes the requested speaker with ``lower()``
    before looking it up in ``talker_config.spk_id``.  Older locally-produced
    full-SFT checkpoints could persist the display name (for example ``PL``)
    instead of the canonical key (``pl``), which made an otherwise valid
    checkpoint fail in both Standard and Faster Qwen inference.  Normalize the
    in-memory registry so old checkpoints remain usable without rewriting them
    on disk.
    """
    inner = getattr(wrapper, "model", wrapper)
    config = getattr(inner, "config", None)
    talker_config = getattr(config, "talker_config", None)
    speaker_ids = getattr(talker_config, "spk_id", None)
    speaker_dialects = getattr(talker_config, "spk_is_dialect", None)
    if not isinstance(speaker_ids, dict) or not isinstance(speaker_dialects, dict):
        return False

    normalized_ids = {str(key).lower(): value for key, value in speaker_ids.items()}
    normalized_dialects = {str(key).lower(): value for key, value in speaker_dialects.items()}
    changed = normalized_ids != speaker_ids or normalized_dialects != speaker_dialects
    if changed:
        speaker_ids.clear()
        speaker_ids.update(normalized_ids)
        speaker_dialects.clear()
        speaker_dialects.update(normalized_dialects)
    if hasattr(inner, "supported_speakers"):
        inner.supported_speakers = speaker_ids.keys()
    return changed


def _first_parameter_device(module):
    """Return a module's first real parameter device without importing torch."""
    if module is None or not hasattr(module, "parameters"):
        return None
    try:
        return next(module.parameters()).device
    except (StopIteration, TypeError, AttributeError):
        return None


def _qwen_core_model(wrapper):
    """Find the official Qwen core through Standard/Faster/PEFT wrappers."""
    pending = [wrapper]
    visited = set()
    while pending:
        candidate = pending.pop(0)
        if candidate is None or id(candidate) in visited:
            continue
        visited.add(id(candidate))
        # FasterQwen also exposes a ``speech_tokenizer`` property, but its
        # public object is only a wrapper.  The official core is identified
        # by the talker (and, for Base, the speaker encoder).
        if hasattr(candidate, "talker") or hasattr(candidate, "speaker_encoder"):
            return candidate
        for name in ("model", "base_model"):
            child = getattr(candidate, name, None)
            if child is not None and id(child) not in visited:
                pending.append(child)
    return getattr(wrapper, "model", wrapper)


def _prepare_qwen_inference_devices(wrapper, context="inference"):
    """Keep Qwen's separate codec and Base speaker encoder beside the talker.

    Qwen's speech tokenizer is not a child of the public wrapper itself.  A
    loader can therefore leave its model on CPU even when the talker and the
    Base speaker encoder are on CUDA.  That is functionally valid, but codec
    encode/decode then becomes the dominant cost for voice cloning and eval.
    This app uses one device for inference, so explicitly co-locate both
    auxiliary networks and report the actual placement.
    """
    try:
        import torch
    except Exception:
        return {"talker": None, "speaker_encoder": None, "speech_tokenizer": None}

    core = _qwen_core_model(wrapper)
    talker = getattr(core, "talker", None)
    speaker_encoder = getattr(core, "speaker_encoder", None)
    speech_tokenizer = getattr(core, "speech_tokenizer", None)
    tokenizer_model = getattr(speech_tokenizer, "model", None)

    target = _first_parameter_device(talker) or _first_parameter_device(core)
    if target is None:
        target = getattr(wrapper, "device", None) or getattr(core, "device", None)
    if target is None:
        target = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    target = torch.device(target)

    def place(module, label):
        if module is None or not hasattr(module, "to"):
            return None
        before = _first_parameter_device(module)
        try:
            if before is not None and before != target:
                module.to(target)
            if hasattr(module, "eval"):
                module.eval()
            return before
        except Exception as exc:
            log(f"[QWEN] {context}: could not move {label} from {before} to {target}: {exc}", level="WARN")
            return before

    speaker_before = place(speaker_encoder, "Base speaker encoder")
    tokenizer_before = place(tokenizer_model, "speech tokenizer")
    if speech_tokenizer is not None:
        # The tokenizer wrapper uses this value to move feature-extractor
        # tensors before encode/decode.  Updating only the wrapper attribute
        # is intentional; the actual network was moved above.
        speech_tokenizer.device = target

    devices = {
        "talker": str(_first_parameter_device(talker) or _first_parameter_device(core) or target),
        "speaker_encoder": str(_first_parameter_device(speaker_encoder) or "not present"),
        "speech_tokenizer": str(_first_parameter_device(tokenizer_model) or "not present"),
    }
    log(
        f"[QWEN] {context} devices: talker={devices['talker']} · "
        f"speaker_encoder={devices['speaker_encoder']} "
        f"(was {speaker_before or 'not present'}) · "
        f"speech_tokenizer={devices['speech_tokenizer']} "
        f"(was {tokenizer_before or 'not present'})"
    )
    if target.type == "cuda" and any(
        value not in {str(target), "not present"} for value in (devices["talker"], devices["speaker_encoder"], devices["speech_tokenizer"])
    ):
        log(
            f"[QWEN] {context}: at least one active inference component is not on {target}; "
            "audio file loading/resampling remains CPU-side, but neural codec and speaker work should be GPU-side.",
            level="WARN",
        )
    return devices


def _repair_qwen_checkpoint_config(model_path):
    """Repair serializer-only fields that the pinned Qwen config rejects.

    ``save_pretrained`` serializes the nested speaker encoder as a generic
    Transformers config and adds ``model_type`` (and occasionally other
    generic fields).  Qwen3TTSSpeakerEncoderConfig in the current qwen-tts
    release has a deliberately strict constructor and accepts only its
    speaker-encoder parameters.  This repair is metadata-only: it never
    changes weights or the model architecture, and it also makes older
    merged checkpoints usable without requiring a new merge.
    """
    path = Path(model_path) if model_path else None
    config_path = path / "config.json" if isinstance(path, Path) else None
    if config_path is None or not config_path.is_file():
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Could not read checkpoint config: {config_path}") from exc
    speaker_config = config.get("speaker_encoder_config")
    if not isinstance(speaker_config, dict):
        return False
    allowed = {
        "mel_dim",
        "enc_dim",
        "enc_channels",
        "enc_kernel_sizes",
        "enc_dilations",
        "enc_attention_channels",
        "enc_res2net_scale",
        "enc_se_channels",
        "sample_rate",
    }
    repaired = {key: value for key, value in speaker_config.items() if key in allowed}
    if repaired == speaker_config:
        return False
    config["speaker_encoder_config"] = repaired
    temporary = config_path.with_name(f".{config_path.name}.repair-{uuid.uuid4().hex[:8]}")
    try:
        temporary.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(config_path)
    except Exception as exc:
        try:
            temporary.unlink(missing_ok=True)
        except Exception:
            pass
        raise RuntimeError(f"Could not repair checkpoint config: {config_path}: {exc}") from exc
    removed = sorted(set(speaker_config) - set(repaired))
    log(f"[QWEN] Repaired checkpoint speaker_encoder_config at {config_path}; removed unsupported fields: {', '.join(removed)}.", level="WARN")
    return True


def _load_qwen_model(choice, adapter=NONE, lora_scale=1.0, attention=ATTENTION_CHOICES[0]):
    global _MODEL, _MODEL_KEY, _MODEL_ADAPTER
    with _MODEL_LOCK:
        cache_key = (str(choice), str(adapter), round(float(lora_scale or 1.0), 4), attention)
        if _MODEL is not None and _MODEL_KEY == cache_key:
            _prepare_qwen_inference_devices(_MODEL, context="cached Standard Qwen")
            return _MODEL
        unload_all_models(reason="switching Qwen model", reset_compiler=False)
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
        except Exception as exc:
            raise RuntimeError("Qwen runtime is not installed. Run install.bat first.") from exc
        model_path = _model_path_for(choice)
        if isinstance(model_path, Path):
            _repair_qwen_checkpoint_config(model_path)
        cuda = bool(torch.cuda.is_available())
        device = "cuda:0" if cuda else "cpu"
        dtype = torch.bfloat16 if cuda else torch.float32
        last_error = None
        for attn in _attention_chain(attention):
            try:
                kwargs = {"device_map": device, "dtype": dtype, "attn_implementation": attn, "low_cpu_mem_usage": True}
                try:
                    model = Qwen3TTSModel.from_pretrained(str(model_path), **kwargs)
                except TypeError as exc:
                    if "torch_dtype" not in str(exc) and "dtype" not in str(exc):
                        raise
                    kwargs.pop("dtype", None)
                    kwargs["torch_dtype"] = dtype
                    model = Qwen3TTSModel.from_pretrained(str(model_path), **kwargs)
                if _normalize_qwen_speaker_registry(model):
                    log("[QWEN] Normalized local speaker registry to lowercase for Qwen generation.")
                log(f"[QWEN] Loaded {model_path} with attention={attn} on {device}.")
                break
            except Exception as exc:
                last_error = exc
                log(f"[QWEN] Attention backend {attn} unavailable: {exc}", level="WARN")
        else:
            raise RuntimeError(f"Qwen model could not load with FlashAttention 2 → SDPA → eager: {last_error}")
        if adapter and adapter != NONE:
            _apply_lora(model, Path(str(adapter)), float(lora_scale or 1.0))
            log(f"[QWEN] Applied LoRA adapter {adapter} at strength {float(lora_scale or 1.0):.2f}.")
        _prepare_qwen_inference_devices(model, context="Standard Qwen")
        _MODEL = model
        _MODEL_KEY = cache_key
        _MODEL_ADAPTER = adapter
        return model


def _fast_engine_reason(choice, mode, adapter=NONE, refs=None):
    """Return a user-facing reason when Faster Qwen must not be selected."""
    info = _catalog_info(choice)
    if not _cuda_available():
        return "Faster Qwen CUDA Graphs requires an available NVIDIA CUDA device."
    if adapter and adapter != NONE:
        return "A separate PEFT/LoRA adapter is kept on the Standard Qwen engine."
    if int(info.get("hz", 12) or 12) != 12:
        return "Faster Qwen CUDA Graphs is validated for the Qwen3-TTS 12 Hz family only."
    if mode == "Base / Voice Clone" and refs is not None and len(refs) != 1:
        return "Faster Qwen voice cloning currently accepts one reference; multiple references use Standard Qwen."
    return None


def _load_faster_qwen_model(choice, attention=ATTENTION_CHOICES[0]):
    """Load the pinned Faster Qwen CUDA-graph wrapper.

    This is intentionally separate from the official wrapper. The pinned
    0.2.x line is compatible with qwen-tts 0.1.x and keeps the public Qwen
    model available for training and PEFT inference.
    """
    global _FAST_MODEL, _FAST_MODEL_KEY
    with _MODEL_LOCK:
        cache_key = (str(choice), "sdpa")
        if _FAST_MODEL is not None and _FAST_MODEL_KEY == cache_key:
            _prepare_qwen_inference_devices(_FAST_MODEL, context="cached Faster Qwen")
            return _FAST_MODEL
        if not _cuda_available():
            raise RuntimeError("Faster Qwen CUDA Graphs requires an NVIDIA CUDA device.")
        unload_all_models(reason="switching to Faster Qwen CUDA Graphs", reset_compiler=False)
        try:
            import torch
            from faster_qwen3_tts import FasterQwen3TTS
        except Exception as exc:
            raise RuntimeError("Faster Qwen CUDA Graphs is not installed. Run install.bat again.") from exc
        model_path = _model_path_for(choice)
        _validate_faster_model_path(choice, model_path)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        # Faster Qwen's validated fast path uses the model's native SDPA. The
        # attention selector remains active for Standard Qwen, while this
        # backend deliberately avoids the unproven FA2 + graph combination.
        try:
            model = FasterQwen3TTS.from_pretrained(
                str(model_path),
                device="cuda:0",
                dtype=dtype,
                attn_implementation="sdpa",
            )
        except TypeError:
            # Keep compatibility with early 0.2.x builds that still used the
            # torch_dtype spelling internally.
            model = FasterQwen3TTS.from_pretrained(
                str(model_path),
                device="cuda:0",
                dtype=dtype,
            )
        if _normalize_qwen_speaker_registry(model.model):
            log("[QWEN] Normalized local speaker registry to lowercase for Faster Qwen generation.")
        _prepare_qwen_inference_devices(model, context="Faster Qwen")
        _FAST_MODEL = model
        _FAST_MODEL_KEY = cache_key
        info = _catalog_info(choice)
        model_note = " (compatible full-SFT checkpoint)" if info.get("mode") == "Trained Custom Voice" else ""
        log(f"[QWEN] Faster CUDA Graphs backend loaded {model_path}{model_note} with native SDPA on cuda:0.")
        return model


def _validate_faster_model_path(choice, model_path):
    """Validate local artifacts before handing a checkpoint to Faster Qwen.

    A complete official SFT checkpoint is a normal Qwen model directory with
    a custom-voice config plus the unchanged 12 Hz speech tokenizer. Faster
    Qwen can capture graphs after loading those weights; it must not receive a
    partial training run or an adapter directory.
    """
    if not isinstance(model_path, Path) or not model_path.is_dir():
        return
    _repair_qwen_checkpoint_config(model_path)
    info = _catalog_info(choice)
    config_path = model_path / "config.json"
    tokenizer_config = model_path / "speech_tokenizer" / "config.json"
    generation_config = model_path / "generation_config.json"
    if not config_path.is_file():
        raise RuntimeError(f"Faster Qwen needs a complete checkpoint with config.json: {model_path}")
    if not tokenizer_config.is_file():
        raise RuntimeError(f"Checkpoint is missing speech_tokenizer/config.json: {model_path}")
    if not generation_config.is_file():
        raise RuntimeError(f"Checkpoint is missing generation_config.json: {model_path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Could not read checkpoint config: {config_path}") from exc
    tokenizer_type = str(config.get("tokenizer_type", "")).lower()
    if info.get("mode") == "Trained Custom Voice" and config.get("tts_model_type") != "custom_voice":
        raise RuntimeError("The selected trained checkpoint is not a complete Qwen CustomVoice SFT model.")
    if tokenizer_type != "qwen3_tts_tokenizer_12hz":
        raise RuntimeError("Faster Qwen CUDA Graphs currently accepts only the complete Qwen 12 Hz tokenizer layout.")


def _icl_reference_with_silence(path: Path, seconds=ICL_TRAILING_SILENCE_SECONDS):
    """Return an audio tuple with trailing silence for a clean ICL boundary."""
    import numpy as np
    data, sr = _audio_read(path, SAMPLE_RATE)
    tail = np.zeros(max(0, int(float(seconds) * sr)), dtype=np.float32)
    return np.concatenate([data, tail]), sr


def _apply_lora(wrapper, adapter_path: Path, scale: float):
    try:
        from peft import PeftModel
    except Exception as exc:
        raise RuntimeError("This adapter needs PEFT. Re-run install.bat to install the LoRA support package.") from exc
    if not (adapter_path / "adapter_config.json").is_file():
        raise RuntimeError(f"LoRA adapter is missing adapter_config.json: {adapter_path}")
    wrapper.model = PeftModel.from_pretrained(wrapper.model, str(adapter_path), is_trainable=False)
    for module in wrapper.model.modules():
        scaling = getattr(module, "scaling", None)
        if isinstance(scaling, dict):
            for key in list(scaling):
                scaling[key] = float(scaling[key]) * float(scale)
        elif isinstance(scaling, (int, float)):
                module.scaling = float(scaling) * float(scale)


def has_packaged_speaker_prompt(model_choice, adapter=NONE):
    """Return whether a complete model or adapter bundles a speaker prompt."""
    return _find_packaged_speaker_prompt(model_choice, adapter) is not None


def _find_packaged_speaker_prompt(model_choice, adapter=NONE):
    candidates = []
    if str(model_choice or "").startswith("local::"):
        candidates.append(Path(str(model_choice)[7:]))
    if adapter and adapter != NONE:
        candidates.append(Path(str(adapter)))
    seen = set()
    for candidate in candidates:
        try:
            candidate = candidate.resolve()
        except OSError:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        prompt_path = candidate / PACKAGED_SPEAKER_PROMPT_FILE
        if prompt_path.is_file():
            return prompt_path
    return None


def _find_packaged_icl_prompt(model_choice, adapter=NONE):
    """Locate the reference-free ICL prompt beside a model or adapter."""
    candidates = []
    if str(model_choice or "").startswith("local::"):
        candidates.append(Path(str(model_choice)[7:]))
    if adapter and adapter != NONE:
        candidates.append(Path(str(adapter)))
    seen = set()
    for candidate in candidates:
        try:
            candidate = candidate.resolve()
        except OSError:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        prompt_path = candidate / PACKAGED_ICL_PROMPT_FILE
        if prompt_path.is_file():
            return prompt_path
    return None


def has_packaged_icl_prompt(model_choice, adapter=NONE):
    """Return whether a trained artifact bundles a complete ICL prompt."""
    return _find_packaged_icl_prompt(model_choice, adapter) is not None


def _load_packaged_speaker_prompt(model_choice, adapter=NONE):
    """Load a reference-free x-vector prompt bundled with a trained artifact.

    The prompt deliberately contains only the learned speaker representation.
    It is not a generation cache: Qwen's Base generation path consumes it as a
    normal ``voice_clone_prompt`` with ICL disabled and no reference audio.
    """
    prompt_path = _find_packaged_speaker_prompt(model_choice, adapter)
    if prompt_path is None:
        return None
    try:
        from safetensors.torch import load_file
        tensors = load_file(str(prompt_path), device="cpu")
        embedding = tensors.get("ref_spk_embedding")
        if embedding is None:
            raise RuntimeError("the packaged prompt has no ref_spk_embedding tensor")
        import torch
        embedding = embedding.detach().to("cpu")
        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)
        if embedding.ndim != 2 or embedding.shape[0] != 1 or not bool(torch.isfinite(embedding).all().item()):
            raise RuntimeError(f"invalid speaker embedding shape {tuple(embedding.shape)}")
        prompt = {
            "ref_code": [None],
            "ref_spk_embedding": [embedding],
            "x_vector_only_mode": [True],
            "icl_mode": [False],
        }
        log(f"[QWEN] Loaded packaged trained speaker prompt: {prompt_path}")
        return prompt
    except Exception as exc:
        log(f"[QWEN] Packaged trained speaker prompt unavailable: {prompt_path} · {exc}", level="WARN")
        return None


def _load_packaged_icl_prompt(model_choice, adapter=NONE):
    """Load a self-contained ICL prompt exported with a trained artifact.

    Unlike the speaker-only prompt, this contains the codec sequence and the
    exact reference transcript required by Qwen's ICL path.  It is therefore
    usable without opening or uploading the original reference audio.
    """
    prompt_path = _find_packaged_icl_prompt(model_choice, adapter)
    if prompt_path is None:
        return None
    try:
        from safetensors.torch import load_file
        import torch

        tensors = load_file(str(prompt_path), device="cpu")
        embedding = tensors.get("ref_spk_embedding")
        ref_code = tensors.get("ref_code")
        metadata_path = prompt_path.with_name(PACKAGED_ICL_PROMPT_METADATA)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        ref_text = str(metadata.get("ref_text") or "").strip()
        if embedding is None or ref_code is None:
            raise RuntimeError("the packaged ICL prompt needs ref_spk_embedding and ref_code tensors")
        if not ref_text:
            raise RuntimeError("the packaged ICL prompt has no reference transcript")
        embedding = embedding.detach().to("cpu")
        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)
        ref_code = ref_code.detach().to("cpu").long()
        if embedding.ndim != 2 or embedding.shape[0] != 1 or not bool(torch.isfinite(embedding).all().item()):
            raise RuntimeError(f"invalid speaker embedding shape {tuple(embedding.shape)}")
        if ref_code.ndim != 2 or ref_code.shape[0] < 1 or ref_code.shape[1] < 1:
            raise RuntimeError(f"invalid reference codec shape {tuple(ref_code.shape)}")
        prompt = {
            "ref_code": ref_code.contiguous(),
            "ref_spk_embedding": embedding[:1].contiguous(),
            "ref_text": ref_text,
            "reference_language": str(metadata.get("reference_language") or "Auto"),
            "source_audio": str(metadata.get("source_audio") or ""),
        }
        log(
            f"[QWEN] Loaded packaged ICL prompt: {prompt_path} · "
            f"{int(ref_code.shape[0])} codec frames · transcript={len(ref_text)} chars"
        )
        return prompt
    except Exception as exc:
        log(f"[QWEN] Packaged ICL prompt unavailable: {prompt_path} · {exc}", level="WARN")
        return None


def _speaker_prompt_from_icl(packaged_icl_prompt):
    """Build the speaker-only view from an ICL package when needed."""
    if not packaged_icl_prompt:
        return None
    return {
        "ref_code": [None],
        "ref_spk_embedding": [packaged_icl_prompt["ref_spk_embedding"]],
        "x_vector_only_mode": [True],
        "icl_mode": [False],
    }


def _remove_runtime_config_dtypes(config):
    """Temporarily remove Transformers runtime dtype fields before saving.

    Qwen's loader propagates the requested ``dtype`` to the root config and
    nested talker configs. Transformers 4.57's diff serializer compares those
    nested values against a default dictionary that does not contain the
    runtime-only key, producing ``KeyError: 'dtype'``. The field is not
    required in a portable checkpoint because the loader chooses the runtime
    dtype again. Return a restoration list so the live object is unchanged.
    """
    removed = []
    visited = set()

    def visit(value):
        if value is None or id(value) in visited:
            return
        visited.add(id(value))
        if hasattr(value, "__dict__"):
            if "dtype" in vars(value):
                removed.append((value, vars(value)["dtype"]))
                delattr(value, "dtype")
            for child in vars(value).values():
                visit(child)
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(config)
    return removed


def _restore_runtime_config_dtypes(removed):
    for config, value in removed:
        try:
            setattr(config, "dtype", value)
        except Exception:
            pass


def merge_lora_for_fast_inference(base_choice, adapter=NONE, lora_scale=1.0, attention=ATTENTION_CHOICES[0]):
    """Merge a PEFT adapter into a complete local 12 Hz Qwen checkpoint.

    Faster Qwen captures graphs over the final model parameters and does not
    apply a separate PEFT adapter at generation time. This operation creates
    a new, self-contained checkpoint; it never overwrites the base model or
    the adapter. The returned ``local::`` choice can be selected immediately
    and used with the Faster Qwen CUDA Graphs engine.
    """
    partial = None
    with _MODEL_LOCK:
        try:
            info = _catalog_info(base_choice)
            if info.get("mode") != "Base / Voice Clone":
                return "Merge requires a 12 Hz Base / Voice Clone model as the LoRA base.", None
            if int(info.get("hz", 12) or 12) != 12:
                return "Merge for Faster Qwen currently supports only the Qwen 12 Hz Base family.", None
            if not adapter or adapter == NONE:
                return "Select a LoRA adapter before merging.", None
            adapter_path = Path(str(adapter))
            if not (adapter_path / "adapter_config.json").is_file():
                return f"LoRA adapter is missing adapter_config.json: {adapter_path}", None
            base_path = _model_path_for(base_choice)
            if not isinstance(base_path, Path) or not base_path.is_dir():
                return "Download the selected base model into the local model library before merging.", None
            _validate_faster_model_path(base_choice, base_path)
            config = json.loads((adapter_path / "adapter_config.json").read_text(encoding="utf-8"))
            base_model_hint = str(config.get("base_model_name_or_path", "")).strip()
            if base_model_hint:
                log(f"[QWEN] Merging LoRA base hint '{base_model_hint}' into {base_path}.")

            destination_parent = FAST_MERGED_MODELS
            destination_parent.mkdir(parents=True, exist_ok=True)
            destination = destination_parent / (
                f"{_safe(base_path.name)}__{_safe(adapter_path.name)}__"
                f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
            )
            partial = destination_parent / f".{destination.name}.partial-{uuid.uuid4().hex[:8]}"
            partial.mkdir(parents=True, exist_ok=False)

            log(f"[QWEN] Preparing LoRA merge for Faster Qwen: {adapter_path} → {destination}.")
            wrapper = _load_qwen_model(base_choice, NONE, 1.0, attention)
            _apply_lora(wrapper, adapter_path, float(lora_scale or 1.0))
            peft_model = wrapper.model
            merge = getattr(peft_model, "merge_and_unload", None)
            if not callable(merge):
                raise RuntimeError("The selected adapter is not a mergeable PEFT LoRA model.")
            merged_model = merge()
            removed_dtypes = _remove_runtime_config_dtypes(getattr(merged_model, "config", None))
            try:
                merged_model.save_pretrained(str(partial), safe_serialization=True, max_shard_size="4GB")
            finally:
                _restore_runtime_config_dtypes(removed_dtypes)
            _repair_qwen_checkpoint_config(partial)
            wrapper.processor.save_pretrained(str(partial))

            speech_tokenizer = base_path / "speech_tokenizer"
            if not (speech_tokenizer / "config.json").is_file():
                raise RuntimeError("The base model is missing speech_tokenizer/config.json.")
            shutil.copytree(speech_tokenizer, partial / "speech_tokenizer")
            generation_config = base_path / "generation_config.json"
            if generation_config.is_file():
                shutil.copy2(generation_config, partial / "generation_config.json")
            else:
                generate_config = getattr(merged_model, "generate_config", None)
                if not isinstance(generate_config, dict):
                    raise RuntimeError("The merged checkpoint is missing generation_config.json.")
                (partial / "generation_config.json").write_text(json.dumps(generate_config, indent=2), encoding="utf-8")
            for prompt_file in (
                PACKAGED_SPEAKER_PROMPT_FILE,
                PACKAGED_SPEAKER_PROMPT_METADATA,
                PACKAGED_ICL_PROMPT_FILE,
                PACKAGED_ICL_PROMPT_METADATA,
            ):
                source_prompt = adapter_path / prompt_file
                if source_prompt.is_file():
                    shutil.copy2(source_prompt, partial / prompt_file)
            (partial / "qwen_easy_fast_merge.json").write_text(json.dumps({
                "training_type": "merged_lora_fast_inference",
                "base_choice": str(base_choice),
                "base_model_path": str(base_path),
                "base_model_hint": base_model_hint,
                "adapter_path": str(adapter_path),
                "lora_scale": float(lora_scale or 1.0),
                "created": datetime.now().isoformat(timespec="seconds"),
            }, indent=2), encoding="utf-8")
            _validate_faster_model_path(f"local::{partial}", partial)
            partial.replace(destination)
            partial = None
            selected = f"local::{destination}"
            log(f"[QWEN] LoRA merge complete. Select {selected} with Faster Qwen CUDA Graphs.")
            play_done_chime()
            return f"Merged LoRA checkpoint created at {destination}. Select it with Faster Qwen CUDA Graphs; the original base and adapter were preserved.", selected
        except Exception as exc:
            import traceback
            log(f"[QWEN] LoRA merge failed: {exc}\n{traceback.format_exc()}", level="ERROR")
            return f"LoRA merge failed: {exc}", None
        finally:
            if partial is not None and partial.exists():
                shutil.rmtree(partial, ignore_errors=True)
            unload_all_models(reason="finished LoRA merge", reset_compiler=False)


def unload_all_models(reason="manual", reset_compiler=True, trim_ram=True):
    global _MODEL, _MODEL_KEY, _MODEL_ADAPTER, _FAST_MODEL, _FAST_MODEL_KEY, _ASR, _ASR_KEY
    with _MODEL_LOCK:
        _MODEL = None
        _MODEL_KEY = None
        _MODEL_ADAPTER = None
        _FAST_MODEL = None
        _FAST_MODEL_KEY = None
        _ASR = None
        _ASR_KEY = None
        _PROMPT_CACHE.clear()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
    return f"Unloaded Qwen models and cleared caches ({reason})."


def get_supported_speakers(choice=None):
    selected_mode = "CustomVoice"
    if choice:
        try:
            selected_mode = _catalog_info(choice).get("mode", "CustomVoice")
            if selected_mode not in {"CustomVoice", "Trained Custom Voice"}:
                return []
        except Exception:
            pass
    result = [] if selected_mode == "Trained Custom Voice" else list(SPEAKERS)
    if choice and str(choice).startswith("local::"):
        path = Path(str(choice)[7:])
        for candidate in (path / "qwen_easy_training.json", path.parent / "qwen_easy_training.json", path / "qwen_easy_lora.json"):
            if candidate.is_file():
                try:
                    speaker = json.loads(candidate.read_text(encoding="utf-8")).get("speaker_name")
                    if speaker and speaker not in result:
                        result.insert(0, speaker)
                except Exception:
                    pass
        if selected_mode == "Trained Custom Voice" and not result:
            try:
                config = json.loads((path / "config.json").read_text(encoding="utf-8"))
                speaker_ids = config.get("talker_config", {}).get("spk_id", {})
                if isinstance(speaker_ids, dict):
                    result.extend(str(name) for name in speaker_ids if str(name).strip())
            except Exception:
                pass
    if selected_mode == "Trained Custom Voice" and not result:
        result.append("speaker_1")
    return result


def split_long_text(text, mode="None"):
    text = str(text or "").strip()
    if not text:
        return []
    if mode == "None":
        return [text]
    if mode == "Lines":
        chunks = [line.strip() for line in text.splitlines() if line.strip()]
    elif mode == "Paragraphs":
        chunks = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    elif mode == "Periods":
        chunks = [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+", text) if part.strip()]
    else:
        chunks = [text]
    return chunks or [text]


def _generation_kwargs(temperature, top_p, top_k, repetition_penalty, max_new_tokens):
    return {"do_sample": True, "temperature": float(temperature), "top_p": float(top_p), "top_k": int(top_k), "repetition_penalty": float(repetition_penalty), "max_new_tokens": int(max_new_tokens)}


def _report_progress(progress, value, desc):
    """Update Gradio progress without allowing a disconnected UI to fail inference."""
    if progress is None:
        return
    try:
        progress(max(0.0, min(1.0, float(value))), desc=desc)
    except Exception:
        pass


def _count_native_codec_tokens(result):
    """Count codec tokens returned by one native Qwen generation call."""
    try:
        codes = result[0]
        if isinstance(codes, (list, tuple)):
            return sum(int(getattr(item, "shape", [0])[0]) for item in codes)
        shape = getattr(codes, "shape", ())
        return int(shape[0]) if shape else 0
    except Exception:
        return 0


def _start_native_generation_meter(active_model, active_engine):
    """Capture Qwen's native codec-token output for one generation call.

    Standard Qwen exposes the final codec sequence internally.  Faster Qwen
    exposes it in its internal ``fast_generate`` return tuple. Neither public
    wrapper exposes a per-token callback, so this records exact totals when
    the native generation call returns.
    """
    meter = {"tokens": 0, "native": False, "timing": None, "restore": None}

    if active_engine == "Faster Qwen CUDA Graphs":
        try:
            import importlib

            generate_module = importlib.import_module("faster_qwen3_tts.generate")
            original_fast_generate = generate_module.fast_generate

            def wrapped_fast_generate(*args, **kwargs):
                result = original_fast_generate(*args, **kwargs)
                meter["tokens"] = _count_native_codec_tokens(result)
                try:
                    meter["timing"] = result[1]
                except Exception:
                    meter["timing"] = None
                meter["native"] = True
                return result

            generate_module.fast_generate = wrapped_fast_generate
            meter["restore"] = lambda: setattr(generate_module, "fast_generate", original_fast_generate)
            return meter
        except Exception as exc:
            log(f"[QWEN] Native token meter unavailable for Faster Qwen: {exc}", level="WARN")
            return meter

    core = getattr(active_model, "model", None)
    original_generate = getattr(core, "generate", None)
    if core is None or original_generate is None:
        return meter

    def wrapped_generate(*args, **kwargs):
        result = original_generate(*args, **kwargs)
        meter["tokens"] = _count_native_codec_tokens(result)
        meter["native"] = True
        return result

    try:
        setattr(core, "generate", wrapped_generate)
        meter["restore"] = lambda: setattr(core, "generate", original_generate)
    except Exception as exc:
        log(f"[QWEN] Native token meter could not attach: {exc}", level="WARN")
    return meter


def _stop_native_generation_meter(meter):
    restore = meter.get("restore") if isinstance(meter, dict) else None
    if restore is not None:
        try:
            restore()
        except Exception as exc:
            log(f"[QWEN] Native token meter cleanup failed: {exc}", level="WARN")


def _estimate_codec_tokens(audio, sample_rate, hz=12):
    """Fallback estimate for an engine that does not expose its token tensor."""
    try:
        return max(0, int(round(len(audio) / max(1, int(sample_rate)) * float(hz))))
    except Exception:
        return 0


def _format_token_metrics(generated, total, elapsed):
    generated = max(0, int(generated or 0))
    total = max(0, int(total or 0))
    elapsed = max(0.0, float(elapsed or 0.0))
    percentage = (generated / total * 100.0) if total else 0.0
    speed = generated / elapsed if elapsed > 0.0 else 0.0
    speed_text = f"{speed:.1f} it/s" if generated and elapsed > 0.0 else "-- it/s"
    return f"codec tokens {generated}/{total} ({percentage:.1f}%) · {speed_text}"


def _as_numpy_audio(audio):
    import numpy as np
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    elif hasattr(audio, "cpu"):
        audio = audio.cpu().numpy()
    return np.asarray(audio, dtype=np.float32).squeeze()


def _resolve_refs(ref_audio, voice_name=None):
    if voice_name and voice_name != NONE:
        audio, transcript, language, _status = load_sample(voice_name)
        transcripts = transcript
        meta_path = _sample_meta(voice_name)
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                values = meta.get("transcripts")
                if isinstance(values, list) and values:
                    transcripts = [str(item or "").strip() for item in values]
            except Exception:
                pass
        # Qwen's official audio normalizer accepts string paths, numpy audio
        # or URL/base64 strings, but not pathlib.Path objects on Windows.
        return [str(path) for path in _safe_audio_value(audio)], transcripts, language
    return [str(path) for path in _safe_audio_value(ref_audio)], "", "Auto"


def _load_legacy_voice_prompt(voice_name, model_choice, refs, clone_texts, x_vector_only):
    """Load a Voice Clone Studio ``.prompt``/``.pt`` prompt when compatible.

    These files contain Qwen ``VoiceClonePromptItem`` objects, not model
    weights.  The cache is accepted only when its audio/transcript hash and
    conditioning mode match the current request.  Multiple-reference entries
    are intentionally left on the normal prompt-building path because the
    legacy cache format is one sample per file.
    """
    if not voice_name or voice_name == NONE or len(refs) != 1:
        return None
    try:
        info = _catalog_info(model_choice)
        model_size_match = re.search(r"(?:0\.6|1\.7)B", str(info.get("size", "")), re.IGNORECASE)
        model_size = model_size_match.group(0) if model_size_match else ""
        if not model_size:
            return None
        prompt_path = _legacy_prompt_path(str(voice_name), model_size)
        if prompt_path is None:
            return None
        audio_path = Path(str(refs[0]))
        if not audio_path.is_file():
            return None
        text = str(clone_texts[0] if clone_texts else "")
        hasher = hashlib.md5()
        with audio_path.open("rb") as handle:
            hasher.update(handle.read())
        hasher.update(text.encode("utf-8"))
        expected_hash = hasher.hexdigest()
        import torch
        cache_data = torch.load(prompt_path, map_location="cpu", weights_only=False)
        if not isinstance(cache_data, dict) or cache_data.get("hash") != expected_hash:
            log(f"[QWEN] Ignoring legacy prompt cache with stale hash: {prompt_path.name}", level="WARN")
            return None
        prompt = cache_data.get("prompt")
        if not isinstance(prompt, (list, tuple)) or not prompt:
            return None
        first = prompt[0]
        cached_x_vector_only = getattr(first, "x_vector_only_mode", None)
        if cached_x_vector_only is not None and bool(cached_x_vector_only) != bool(x_vector_only):
            log(f"[QWEN] Ignoring legacy prompt cache with incompatible conditioning mode: {prompt_path.name}", level="WARN")
            return None
        log(f"[QWEN] Loaded legacy voice prompt cache: {prompt_path.name}")
        return list(prompt)
    except Exception as exc:
        log(f"[QWEN] Legacy voice prompt cache unavailable ({voice_name}): {exc}", level="WARN")
        return None


@_inference_job
def synthesize(model_choice, mode, text, language, speaker, instruct, voice_name, ref_audio, ref_text, x_vector_only, adapter, lora_scale, temperature, top_p, top_k, repetition_penalty, max_new_tokens, chunk_mode, chunk_gap, attention, seed=None, progress=None, engine=DEFAULT_INFERENCE_ENGINE, play_chime=True):
    if not str(text or "").strip():
        return None, "Enter text to synthesize.", seed
    chosen_seed = apply_generation_seed(seed) or random_seed_value()
    info = _catalog_info(model_choice)
    selected_mode = mode or info.get("mode", "Base / Voice Clone")
    started_at = time.perf_counter()
    if selected_mode != "Base / Voice Clone" and adapter and adapter != NONE:
        log(f"[QWEN] Ignoring LoRA adapter for unsupported workflow {selected_mode}.", level="WARN")
        adapter = NONE
        lora_scale = 1.0
    if selected_mode == "VoiceDesign" and not str(instruct or "").strip():
        return None, "VoiceDesign needs a natural-language voice description.", chosen_seed
    packaged_speaker_prompt = _load_packaged_speaker_prompt(model_choice, adapter)
    packaged_icl_prompt = _load_packaged_icl_prompt(model_choice, adapter)
    refs, saved_ref_text, _saved_lang = _resolve_refs(ref_audio, voice_name)
    packaged_conditioning = None
    packaged_reference_free = selected_mode == "Base / Voice Clone" and not refs and (
        packaged_speaker_prompt is not None or packaged_icl_prompt is not None
    )
    if selected_mode == "Base / Voice Clone" and not refs:
        if not bool(x_vector_only) and packaged_icl_prompt is not None:
            packaged_conditioning = "ICL"
            log("[QWEN] Using packaged ICL prompt; reference audio and transcript are not required for this request.")
        elif packaged_speaker_prompt is not None:
            packaged_conditioning = "Speaker-only"
            x_vector_only = True
            log("[QWEN] Using packaged trained speaker prompt; reference audio is not required for this request.")
        elif packaged_icl_prompt is not None:
            packaged_speaker_prompt = _speaker_prompt_from_icl(packaged_icl_prompt)
            packaged_conditioning = "Speaker-only"
            x_vector_only = True
            log("[QWEN] Using the speaker-only view of the packaged ICL prompt; reference audio is not required.")
        else:
            return None, "Base / Voice Clone needs a reference audio file, Voice Library entry, or a trained artifact with a packaged prompt.", chosen_seed
    clone_value = ref_text if ref_text not in (None, "") else saved_ref_text
    if packaged_conditioning == "ICL" and packaged_icl_prompt is not None:
        # The packaged codec sequence and transcript are one inseparable ICL
        # pair. Never combine it with a manually typed transcript that belongs
        # to a missing external audio file.
        clone_value = packaged_icl_prompt.get("ref_text", "")
    if isinstance(clone_value, (list, tuple)):
        clone_texts = [str(item or "").strip() for item in clone_value]
        clone_text = clone_texts[0] if len(clone_texts) == 1 else clone_texts
    else:
        clone_text = str(clone_value or "").strip()
        clone_texts = [clone_text] * len(refs)
    if selected_mode == "Base / Voice Clone" and not bool(x_vector_only) and (not clone_text or any(not item for item in clone_texts[:len(refs)])):
        return None, "ICL cloning needs the exact reference transcript. Enable speaker-only mode if no transcript is available.", chosen_seed
    try:
        log(f"[QWEN] Inference started: mode={selected_mode} engine={engine} max_new_tokens={int(max_new_tokens)}.")
        _report_progress(progress, 0.0, "Loading Qwen model")
        requested_engine = engine if engine in INFERENCE_ENGINE_CHOICES else DEFAULT_INFERENCE_ENGINE
        fallback_reason = _fast_engine_reason(model_choice, selected_mode, adapter, refs) if requested_engine == "Faster Qwen CUDA Graphs" else None
        effective_engine = requested_engine
        if fallback_reason:
            effective_engine = DEFAULT_INFERENCE_ENGINE
            log(f"[QWEN] Faster CUDA Graphs fallback: {fallback_reason}", level="WARN")
        if effective_engine == "Faster Qwen CUDA Graphs":
            try:
                model = _load_faster_qwen_model(model_choice, attention)
            except Exception as fast_exc:
                # Graph capture can fail for a particular driver, GPU, or
                # model cache even when CUDA itself is available. Keep the
                # request usable by retrying on the official implementation.
                fallback_reason = f"Faster Qwen CUDA Graphs could not initialize: {fast_exc}"
                effective_engine = DEFAULT_INFERENCE_ENGINE
                log(f"[QWEN] Faster CUDA Graphs initialization failed; using Standard Qwen: {fast_exc}", level="WARN")
                model = _load_qwen_model(model_choice, adapter, lora_scale, attention)
        else:
            model = _load_qwen_model(model_choice, adapter, lora_scale, attention)
        chunks = split_long_text(text, chunk_mode)
        log(f"[QWEN] Model ready; {len(chunks)} chunk(s), {len(str(text))} input characters.")
        _report_progress(progress, 0.08, "Preparing reference conditioning")
        import numpy as np
        def _build_standard_prompt(active_model):
            if selected_mode != "Base / Voice Clone":
                return None
            if packaged_conditioning == "Speaker-only":
                return packaged_speaker_prompt
            if packaged_conditioning == "ICL" and packaged_icl_prompt is not None:
                try:
                    from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem
                    return [VoiceClonePromptItem(
                        ref_code=packaged_icl_prompt["ref_code"],
                        ref_spk_embedding=packaged_icl_prompt["ref_spk_embedding"].squeeze(0),
                        x_vector_only_mode=False,
                        icl_mode=True,
                        ref_text=packaged_icl_prompt["ref_text"],
                    )]
                except Exception as exc:
                    raise RuntimeError(f"Could not build the packaged ICL prompt: {exc}") from exc
            conditioning_started_at = time.perf_counter()
            prompt_key = (str(model_choice), str(adapter), tuple(str(p) for p in refs), tuple(clone_texts), bool(x_vector_only), ICL_TRAILING_SILENCE_SECONDS if not x_vector_only else 0.0)
            cached_prompt = _PROMPT_CACHE.get(prompt_key)
            if cached_prompt is not None:
                log(
                    f"[QWEN] Reference conditioning cache hit: mode={'speaker-only' if x_vector_only else 'ICL'} · "
                    f"references={len(refs)} · elapsed={time.perf_counter() - conditioning_started_at:.3f}s."
                )
                return cached_prompt
            legacy_prompt = _load_legacy_voice_prompt(
                voice_name,
                model_choice,
                refs,
                clone_texts,
                bool(x_vector_only),
            )
            if legacy_prompt is not None:
                _PROMPT_CACHE[prompt_key] = legacy_prompt
                log(
                    f"[QWEN] Reference conditioning cache hit (disk): mode={'speaker-only' if x_vector_only else 'ICL'} · "
                    f"references={len(refs)} · elapsed={time.perf_counter() - conditioning_started_at:.3f}s."
                )
                return legacy_prompt
            if x_vector_only:
                prompt_audio = refs if len(refs) > 1 else refs[0]
            else:
                prompt_audio = [_icl_reference_with_silence(path) for path in refs] if len(refs) > 1 else _icl_reference_with_silence(refs[0])
            prompt_text = clone_texts if len(refs) > 1 else clone_text
            log(
                f"[QWEN] Reference conditioning started: mode={'speaker-only' if x_vector_only else 'ICL'} · "
                f"references={len(refs)} · audio encoder and speaker encoder will run before generation."
            )
            cached_prompt = active_model.create_voice_clone_prompt(ref_audio=prompt_audio, ref_text=prompt_text, x_vector_only_mode=bool(x_vector_only))
            _PROMPT_CACHE[prompt_key] = cached_prompt
            log(
                f"[QWEN] Reference conditioning complete: mode={'speaker-only' if x_vector_only else 'ICL'} · "
                f"references={len(refs)} · elapsed={time.perf_counter() - conditioning_started_at:.2f}s."
            )
            return cached_prompt

        # Faster Qwen can consume the same precomputed VoiceClonePromptItem
        # list.  Use a legacy disk cache when available; otherwise it keeps
        # its normal raw-reference extraction path.
        if effective_engine == "Faster Qwen CUDA Graphs":
            if packaged_conditioning == "Speaker-only":
                prompt = packaged_speaker_prompt
            elif packaged_conditioning == "ICL" and packaged_icl_prompt is not None:
                prompt = {
                    "ref_code": [packaged_icl_prompt["ref_code"]],
                    "ref_spk_embedding": [packaged_icl_prompt["ref_spk_embedding"]],
                    "x_vector_only_mode": [False],
                    "icl_mode": [True],
                }
            else:
                prompt = _load_legacy_voice_prompt(
                    voice_name,
                    model_choice,
                    refs,
                    clone_texts,
                    bool(x_vector_only),
                )
        else:
            prompt = _build_standard_prompt(model)
        _report_progress(progress, 0.12, "Starting speech generation")

        def _render_chunks(active_model, active_engine, active_prompt):
            rendered_chunks = []
            rendered_sr = SAMPLE_RATE
            total_token_limit = max(1, int(max_new_tokens)) * max(1, len(chunks))
            generated_tokens = 0
            generation_started_at = time.perf_counter()
            log(f"[QWEN] Speech generation started: {_format_token_metrics(0, total_token_limit, 0.0)}.")
            for index, chunk in enumerate(chunks, start=1):
                if _INFERENCE_CANCEL.is_set():
                    raise RuntimeError("Inference stopped between chunks.")
                chunk_started_at = time.perf_counter()
                chunk_start_progress = 0.12 + (index - 1) / max(1, len(chunks)) * 0.82
                chunk_end_progress = 0.12 + index / max(1, len(chunks)) * 0.82
                _report_progress(
                    progress,
                    chunk_start_progress,
                    f"Generating chunk {index}/{len(chunks)} · {_format_token_metrics(generated_tokens, total_token_limit, 0.0)}",
                )
                log(f"[QWEN] Chunk {index}/{len(chunks)} started: {len(chunk)} characters; token limit={int(max_new_tokens)}.")
                kwargs = _generation_kwargs(temperature, top_p, top_k, repetition_penalty, max_new_tokens)
                meter = _start_native_generation_meter(active_model, active_engine)
                try:
                    if selected_mode == "VoiceDesign":
                        wavs, rendered_sr = active_model.generate_voice_design(text=chunk, instruct=str(instruct).strip(), language=language or "Auto", non_streaming_mode=True, **kwargs)
                    elif selected_mode in {"CustomVoice", "Trained Custom Voice"}:
                        chosen_speaker = speaker if speaker and speaker != NONE else (get_supported_speakers(model_choice)[0])
                        if info.get("size") == "0.6B":
                            kwargs.pop("repetition_penalty", None)
                        wavs, rendered_sr = active_model.generate_custom_voice(text=chunk, speaker=chosen_speaker, language=language or "Auto", instruct=str(instruct or "").strip() or None, **kwargs)
                    elif active_engine == "Faster Qwen CUDA Graphs":
                        # Prefer a validated precomputed prompt; otherwise pass
                        # raw reference audio so Faster Qwen can apply its
                        # validated 0.5 s ICL boundary silence. Most importantly,
                        # xvec_only is explicit and never inferred by the engine.
                        fast_clone_kwargs = {
                            "text": chunk,
                            "language": language or "Auto",
                            "ref_text": clone_text,
                            "xvec_only": bool(x_vector_only),
                            "non_streaming_mode": False,
                            "append_silence": not bool(x_vector_only),
                            **kwargs,
                        }
                        if active_prompt is not None:
                            # A precomputed prompt already contains the speaker
                            # embedding and (for ICL) codec reference tokens.
                            fast_clone_kwargs["voice_clone_prompt"] = active_prompt
                        else:
                            fast_clone_kwargs["ref_audio"] = str(refs[0])
                        wavs, rendered_sr = active_model.generate_voice_clone(**fast_clone_kwargs)
                    else:
                        wavs, rendered_sr = active_model.generate_voice_clone(text=chunk, language=language or "Auto", voice_clone_prompt=active_prompt, **kwargs)
                finally:
                    _stop_native_generation_meter(meter)
                rendered_audio = _as_numpy_audio(wavs[0])
                rendered_chunks.append(rendered_audio)
                chunk_seconds = len(rendered_audio) / max(1, int(rendered_sr))
                elapsed = time.perf_counter() - chunk_started_at
                chunk_tokens = int(meter.get("tokens") or 0)
                if not chunk_tokens:
                    chunk_tokens = _estimate_codec_tokens(rendered_audio, rendered_sr, info.get("hz", 12))
                generated_tokens += chunk_tokens
                metrics = _format_token_metrics(generated_tokens, total_token_limit, time.perf_counter() - generation_started_at)
                log(
                    f"[QWEN] Chunk {index}/{len(chunks)} complete in {elapsed:.1f}s; audio={chunk_seconds:.1f}s; "
                    f"{metrics}; native_token_count={'yes' if meter.get('native') else 'estimated'}."
                )
                _report_progress(progress, chunk_end_progress, f"Chunk {index}/{len(chunks)} complete · {metrics}")
            return rendered_chunks, rendered_sr, generated_tokens, total_token_limit, time.perf_counter() - generation_started_at

        try:
            rendered, sr, generated_tokens, total_token_limit, generation_elapsed = _render_chunks(model, effective_engine, prompt)
        except Exception as generation_exc:
            if _INFERENCE_CANCEL.is_set():
                return None, "Inference stopped between chunks.", chosen_seed
            if effective_engine != "Faster Qwen CUDA Graphs":
                raise
            # A graph may capture successfully and still reject a particular
            # shape or driver state at runtime. Restart the complete request
            # on the official path so output is never mixed across engines.
            fallback_reason = f"Faster Qwen CUDA Graphs failed during generation: {generation_exc}"
            effective_engine = DEFAULT_INFERENCE_ENGINE
            log(f"[QWEN] Faster CUDA Graphs generation failed; restarting on Standard Qwen: {generation_exc}", level="WARN")
            model = _load_qwen_model(model_choice, adapter, lora_scale, attention)
            prompt = _build_standard_prompt(model)
            rendered, sr, generated_tokens, total_token_limit, generation_elapsed = _render_chunks(model, effective_engine, prompt)
        silence = np.zeros(max(0, int(float(chunk_gap or 0) * sr)), dtype=np.float32)
        joined = rendered[0] if len(rendered) == 1 else np.concatenate([part if i == len(rendered) - 1 else np.concatenate([part, silence]) for i, part in enumerate(rendered)])
        folder = OUTPUTS / _safe(selected_mode.replace("/", "_"))
        name = f"{datetime.now():%Y%m%d_%H%M%S}_{_safe(str(text)[:32])}_{uuid.uuid4().hex[:6]}.wav"
        output = folder / name
        _audio_write(output, joined, int(sr))
        elapsed = time.perf_counter() - started_at
        audio_seconds = len(joined) / max(1, int(sr))
        log(f"[QWEN] Generated {output.name} using {info['id']} · {selected_mode} · {effective_engine} · elapsed={elapsed:.1f}s · audio={audio_seconds:.1f}s.")
        token_metrics = _format_token_metrics(generated_tokens, total_token_limit, generation_elapsed)
        log(f"[QWEN] Native generation metrics: {token_metrics}.")
        _report_progress(progress, 1.0, f"Generation complete · {token_metrics} · elapsed {elapsed:.1f}s")
        if play_chime:
            play_done_chime()
        status = f"Generated with {selected_mode} · {info['id']} · {effective_engine} · {elapsed:.1f}s for {audio_seconds:.1f}s audio · {token_metrics} · seed {chosen_seed}."
        if packaged_reference_free:
            status += f" Packaged trained {packaged_conditioning or 'conditioning'} prompt used; no reference audio or transcript was required."
        if fallback_reason:
            status += f" Standard Qwen fallback: {fallback_reason}"
        return str(output), status, chosen_seed
    except Exception as exc:
        log(f"Qwen inference failed: {exc}", level="ERROR")
        return None, f"Qwen inference failed: {exc}", chosen_seed


@_inference_job
def generate_dialogue(model_choice, mode, language, speaker, instruct, adapter, lora_scale, temperature, top_p, top_k, repetition_penalty, max_new_tokens, chunk_mode, chunk_gap, silence, seed, fixed_seed, row_count, *rows, attention=ATTENTION_CHOICES[0], progress=None, engine=DEFAULT_INFERENCE_ENGINE):
    count = max(1, int(row_count or 1))
    # The UI always sends every visible row value first and every row text
    # second, including hidden rows.  Fish and XTTS use the same fixed-width
    # split and only apply the active row count afterward.  Splitting at
    # ``count`` would make row 3's dropdown become row 1's text whenever the
    # count was below MAX_DIALOGUE.
    row_width = max(1, len(rows) // 2)
    voices = list(rows[:row_width])
    texts = list(rows[row_width:row_width * 2])
    fragments = []
    actual_seed = apply_generation_seed(seed) if fixed_seed else random_seed_value()
    turns = [
        (index, voice, text)
        for index, (voice, text) in enumerate(zip(voices[:count], texts[:count]), start=1)
        if str(text or "").strip()
    ]
    started_at = time.perf_counter()
    total_turns = len(turns)
    log(f"[QWEN] Dialogue started: {total_turns} turn(s), engine={engine}, max_new_tokens={int(max_new_tokens)}.")
    _report_progress(progress, 0.0, "Preparing dialogue")
    for position, (source_index, voice, text) in enumerate(turns, start=1):
        row_voice = voice if voice not in (None, NONE) else (speaker if mode != "Base / Voice Clone" else NONE)
        saved_voice = row_voice if mode == "Base / Voice Clone" else NONE
        if mode == "Base / Voice Clone":
            if saved_voice == NONE or saved_voice == PACKAGED_SPEAKER_CHOICE or saved_voice not in get_sample_choices():
                return None, f"Dialogue turn {source_index} needs a valid Voice Library reference. Packaged trained prompts are available only in Single Inference.", actual_seed
        direct_audio = None
        direct_text = ""
        log(f"[QWEN] Dialogue turn {position}/{total_turns} started (source row {source_index}).")
        def turn_progress(value, desc=None, turn_index=position):
            fraction = ((turn_index - 1) + float(value or 0.0)) / max(1, total_turns)
            _report_progress(progress, fraction, f"Turn {turn_index}/{total_turns}: {desc or 'generating'}")
        audio, status, actual_seed = synthesize(model_choice, mode, text, language, row_voice, instruct, saved_voice, direct_audio, direct_text, False, adapter, lora_scale, temperature, top_p, top_k, repetition_penalty, max_new_tokens, chunk_mode, chunk_gap, attention, actual_seed, turn_progress, engine=engine, play_chime=False)
        if not audio:
            return None, status, actual_seed
        data, sr = _audio_read(Path(audio), SAMPLE_RATE)
        fragments.append(data)
        if position < total_turns:
            import numpy as np
            fragments.append(np.zeros(int(float(silence or 0) * sr), dtype=np.float32))
    if not fragments:
        return None, "Add at least one non-empty dialogue turn.", actual_seed
    import numpy as np
    joined = np.concatenate(fragments)
    output = OUTPUTS / "Dialogue" / f"{datetime.now():%Y%m%d_%H%M%S}_dialogue_{uuid.uuid4().hex[:6]}.wav"
    _audio_write(output, joined, SAMPLE_RATE)
    elapsed = time.perf_counter() - started_at
    audio_seconds = len(joined) / SAMPLE_RATE
    log(f"[QWEN] Dialogue generated {output.name} · turns={total_turns} · elapsed={elapsed:.1f}s · audio={audio_seconds:.1f}s.")
    _report_progress(progress, 1.0, f"Dialogue complete · {elapsed:.1f}s")
    play_done_chime()
    return str(output), f"Dialogue generated with {total_turns} turn(s) · {elapsed:.1f}s for {audio_seconds:.1f}s audio.", actual_seed


def _load_asr(model_name):
    global _ASR, _ASR_KEY
    key = whisper_model_id(model_name)
    if _ASR is not None and _ASR_KEY == key:
        return _ASR
    try:
        from faster_whisper import WhisperModel
        device = "cuda" if _cuda_available() else "cpu"
        compute = "float16" if device == "cuda" else "int8"
        _ASR = WhisperModel(key, device=device, compute_type=compute)
        _ASR_KEY = key
        return _ASR
    except Exception as exc:
        raise RuntimeError(f"Faster-Whisper is unavailable or could not load '{key}': {exc}") from exc


def transcribe_only(audio_path, model_name="small", language="Auto", batch_size=4, progress=None):
    paths = _safe_audio_value(audio_path)
    if not paths:
        return "", "Choose an audio file first."
    try:
        model = _load_asr(model_name)
        segments, _info = model.transcribe(str(paths[0]), language=None if language in (None, "Auto") else language.lower(), beam_size=5)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        play_done_chime()
        return text, f"Transcribed {paths[0].name}."
    except Exception as exc:
        return "", f"Transcription failed: {exc}"


def list_datasets():
    return sorted([p.name for p in DATASETS.iterdir() if p.is_dir() and (p / "train_raw.jsonl").is_file()], key=str.casefold)


def _dataset_paths(name):
    root = DATASETS / _safe(name)
    return root, root / "train_raw.jsonl", root / "eval_raw.jsonl", root / "train_with_codes.jsonl", root / "eval_with_codes.jsonl"


def prepare_dataset(source_folder, dataset_name, language="Auto", whisper_model="small", batch_size=4, progress=None, play_chime=True):
    if not source_folder or not dataset_name:
        return "Choose a source folder and Dataset Project first.", "", ""
    source = Path(str(source_folder))
    if not source.is_dir():
        return f"Source folder not found: {source}", "", ""
    begin_aux_job()
    root, train_path, eval_path, _train_codes, _eval_codes = _dataset_paths(dataset_name)
    wavs = root / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    audio_files = sorted([p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in {".wav", ".flac", ".mp3", ".m4a", ".ogg"}], key=lambda p: str(p).casefold())
    if not audio_files:
        return "No supported audio files were found.", "", ""
    records = []
    asr = None
    for index, audio in enumerate(audio_files, start=1):
        if _CANCEL_AUX.is_set():
            return "Dataset preparation stopped.", "", ""
        target = wavs / f"{index:05d}.wav"
        try:
            data, sr = _audio_read(audio, SAMPLE_RATE)
            _audio_write(target, data, sr)
        except Exception as exc:
            log(f"Skipping {audio.name}: {exc}", level="WARN")
            continue
        sidecar = _sidecar_for(audio)
        transcript = sidecar.read_text(encoding="utf-8", errors="ignore").strip() if sidecar else ""
        if not transcript:
            if asr is None:
                asr = _load_asr(whisper_model_id(whisper_model))
            segments, _info = asr.transcribe(str(target), language=None if language == "Auto" else str(language).lower())
            transcript = " ".join(segment.text.strip() for segment in segments).strip()
        if transcript:
            records.append({"audio": str(target.resolve()), "text": transcript, "ref_audio": str(target.resolve()), "language": language})
        if progress:
            progress(index / len(audio_files), desc=f"Preparing {index}/{len(audio_files)}")
    if not records:
        return "No usable audio/transcript pairs were created.", "", ""
    split = max(1, int(len(records) * 0.1)) if len(records) >= 5 else 0
    eval_records = records[:split]
    train_records = records[split:] or records
    for path, values in ((train_path, train_records), (eval_path, eval_records)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in values) + "\n", encoding="utf-8")
    manifest = {"dataset": _safe(dataset_name), "language": language, "sample_count": len(records), "train_count": len(train_records), "eval_count": len(eval_records), "reference_audio": records[0]["ref_audio"], "created_at": datetime.now().isoformat(), "source_folder": str(source.resolve())}
    (root / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"[DATASET] Prepared {len(records)} item(s) for {dataset_name}; train={len(train_records)}, eval={len(eval_records)}.")
    if play_chime:
        play_done_chime()
    return f"Prepared dataset '{_safe(dataset_name)}': {len(records)} item(s), {len(train_records)} train / {len(eval_records)} eval.", str(train_path), str(eval_path)


def extract_audio_codes(dataset_name, tokenizer_model=DEFAULT_TOKENIZER_MODEL, device="auto", progress=None, play_chime=True):
    root, train_raw, eval_raw, train_codes, eval_codes = _dataset_paths(dataset_name)
    if not train_raw.is_file():
        return "Prepare the dataset before extracting Qwen audio codes."
    if "25Hz" in tokenizer_model or (root / "dataset_manifest.json").is_file() and "25" in tokenizer_model:
        return "The vendored official Qwen fine-tuner currently supports the 12 Hz tokenizer path only. Use the 12 Hz Base model for training."
    selected_device = "cuda:0" if device in (None, "", "auto") and _cuda_available() else (device if device not in (None, "", "auto") else "cpu")
    command = [sys.executable, str(FINETUNING / "prepare_data.py"), "--device", selected_device, "--tokenizer_model_path", tokenizer_model, "--input_jsonl", str(train_raw), "--output_jsonl", str(train_codes)]
    try:
        log(f"[DATASET] Extracting Qwen audio codes with {tokenizer_model}.")
        result = subprocess.run(command, cwd=str(FINETUNING), text=True, capture_output=True)
        if result.returncode:
            return f"Audio-code extraction failed: {result.stderr[-1800:]}"
        if eval_raw.is_file() and eval_raw.read_text(encoding="utf-8").strip():
            eval_command = command[:-3] + [str(eval_raw), "--output_jsonl", str(eval_codes)]
            result = subprocess.run(eval_command, cwd=str(FINETUNING), text=True, capture_output=True)
            if result.returncode:
                return f"Train codes completed, eval codes failed: {result.stderr[-1800:]}"
        if play_chime:
            play_done_chime()
        return f"Extracted Qwen audio codes: {train_codes.name}."
    except Exception as exc:
        return f"Audio-code extraction could not start: {exc}"


def _audio_duration_seconds(path):
    """Read audio metadata without decoding the waveform for AutoTune."""
    if not path:
        return 0.0
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return float(info.frames) / max(1, int(info.samplerate))
    except Exception:
        return 0.0


def _percentile(values, fraction):
    ordered = sorted(float(value) for value in values if value is not None)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * max(0.0, min(1.0, float(fraction)))
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _training_file_stats(path):
    """Collect dataset shape statistics used by the full Qwen AutoTune profile."""
    result = {
        "samples": 0,
        "seconds": 0.0,
        "durations": [],
        "text_chars": [],
        "code_frames": [],
    }
    if not path.is_file():
        return result
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            try:
                item = json.loads(raw)
            except Exception:
                continue
            result["samples"] += 1
            text = str(item.get("text") or "").strip()
            if text:
                result["text_chars"].append(len(text))
            duration = _audio_duration_seconds(item.get("audio"))
            codes = item.get("audio_codes")
            frames = len(codes) if isinstance(codes, list) and codes else 0
            if not duration and frames:
                duration = frames / 12.0
            if duration:
                result["durations"].append(duration)
                result["seconds"] += duration
            if frames:
                result["code_frames"].append(frames)
    return result


def dataset_stats(dataset_name):
    root, train_raw, eval_raw, train_codes, eval_codes = _dataset_paths(dataset_name)
    train_path = train_codes if train_codes.is_file() else train_raw
    eval_path = eval_codes if eval_codes.is_file() else eval_raw
    train = _training_file_stats(train_path)
    evaluation = _training_file_stats(eval_path)
    durations = train["durations"]
    text_chars = train["text_chars"]
    code_frames = train["code_frames"]
    return {
        "dataset": dataset_name,
        "train": train["samples"],
        "eval": evaluation["samples"],
        "seconds": train["seconds"],
        "hours": train["seconds"] / 3600.0,
        "mean_seconds": train["seconds"] / max(1, len(durations)),
        "p95_seconds": _percentile(durations, 0.95),
        "max_seconds": max(durations, default=0.0),
        "mean_text_chars": sum(text_chars) / max(1, len(text_chars)),
        "p95_text_chars": _percentile(text_chars, 0.95),
        "mean_code_frames": sum(code_frames) / max(1, len(code_frames)),
        "p95_code_frames": _percentile(code_frames, 0.95),
        "max_code_frames": max(code_frames, default=0),
        "codes_ready": train_codes.is_file(),
    }


def autotune_training(dataset_name, base_choice, training_mode, hardware_preset):
    """Choose a complete, dataset-aware profile for every supported Qwen HP.

    The worker uses AdamW with a conservative warmup/cosine schedule. The
    profile tunes model/runtime attention, epochs, batch/accumulation, learning
    rate scheduling, checkpoint cadence, clipping, weight decay and PEFT
    rank/alpha/dropout. Audio duration, code-frame length, transcript length
    and train/eval counts all influence the result.
    """
    stats = dataset_stats(dataset_name)
    if not stats["train"]:
        raise ValueError("Prepare a dataset before using AutoTune.")
    info = _catalog_info(base_choice)
    size = str(info.get("size", "1.7B"))
    is_small = "0.6" in size.lower().replace("b", ".")
    preset = str(hardware_preset or TRAINING_HARDWARE_PRESETS[2])
    vram_match = re.search(r"(\d+)", preset)
    vram_gb = int(vram_match.group(1)) if vram_match else 24
    if not is_small and vram_gb < 24:
        raise ValueError("The 1.7B Base worker needs the 24 GB or 32 GB+ preset in this GUI. Select the 0.6B Base model for a smaller budget.")
    sample_count = max(1, int(stats["train"]))
    hours = max(stats["hours"], sample_count * max(stats["mean_seconds"], 1.0) / 3600.0)
    p95_frames = max(stats["p95_code_frames"], stats["p95_seconds"] * 12.0)
    long_audio = p95_frames >= 12.0 * 12.0
    text_complexity = max(1.0, stats["p95_text_chars"] / 120.0)
    sequence_complexity = max(1.0, p95_frames / (12.0 * 15.0), text_complexity)

    # Batch is limited by model/VRAM first, then by the 95th-percentile
    # sequence length so one unusually long item does not cause an OOM.
    batch_size = 2 if is_small and vram_gb >= 24 else 1
    if is_small and vram_gb >= 32 and not long_audio and sequence_complexity <= 1.25:
        batch_size = 3
    if not is_small or long_audio or sequence_complexity > 1.5:
        batch_size = 1
    target_effective_batch = 8 if is_small else 4
    if hours >= 4.0:
        target_effective_batch *= 2
    grad_accum = max(1, min(32, int(math.ceil(target_effective_batch / batch_size))))
    if vram_gb <= 12:
        grad_accum = max(grad_accum, 8)
    elif vram_gb <= 16:
        grad_accum = max(grad_accum, 6)

    # Small datasets need more passes; longer and larger datasets need fewer
    # passes to avoid replaying the same speaker material excessively.
    if hours < 0.25 or sample_count < 20:
        epochs = 10 if is_small else 8
    elif hours < 1.0 or sample_count < 60:
        epochs = 8 if is_small else 6
    elif hours < 4.0 or sample_count < 240:
        epochs = 5 if is_small else 4
    elif hours < 12.0:
        epochs = 4 if is_small else 3
    else:
        epochs = 3 if is_small else 2

    lora_mode = training_mode == "PEFT LoRA adapter (experimental)"
    # Qwen's 12 Hz autoregressive codec stack is much more sensitive than a
    # conventional text LM. Keep AutoTune in the conservative range that
    # avoids the rapid bf16/FlashAttention loss explosion seen with 1e-5–2e-5.
    base_lr = 1e-6 if lora_mode else 2e-6
    data_lr_scale = max(0.25, min(1.0, math.sqrt(max(hours, 0.25))))
    if long_audio:
        data_lr_scale *= 0.8
    learning_rate = base_lr * data_lr_scale
    if hours < 0.5:
        weight_decay = 0.05
    elif hours >= 8.0:
        weight_decay = 0.005
    else:
        weight_decay = 0.01
    max_grad_norm = 0.75 if long_audio or sequence_complexity > 1.75 else 1.0
    if lora_mode:
        if hours < 0.5 or sequence_complexity < 1.25:
            rank = 8
        elif hours < 4.0 or sequence_complexity < 2.0:
            rank = 16
        else:
            rank = 32
        alpha = rank * 2
        dropout = 0.10 if hours < 0.5 else (0.05 if hours < 4.0 else 0.02)
    else:
        rank, alpha, dropout = 16, 32, 0.05
    # Checkpoint/evaluation cadence is exposed in complete epochs. The worker
    # still reports optimizer steps for loss/progress, but the user-facing
    # schedule must not depend on batch size or gradient accumulation.
    micro_batches_per_epoch = max(1, int(math.ceil(sample_count / batch_size)))
    optimizer_steps_per_epoch = max(1, int(math.ceil(micro_batches_per_epoch / grad_accum)))
    total_optimizer_steps = max(1, optimizer_steps_per_epoch * epochs)
    epoch_cadence = max(1, min(epochs, int(math.ceil(epochs / 4.0))))

    return {
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "learning_rate": learning_rate,
        "lr_schedule": "warmup_cosine",
        "warmup_ratio": 0.05,
        "min_lr_ratio": 0.20,
        "save_every_epochs": epoch_cadence,
        "eval_every_epochs": epoch_cadence,
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "max_grad_norm": max_grad_norm,
        "training_seed": 1234,
        "attention": TRAINING_ATTENTION_CHOICES[0],
        "summary": (
            f"AutoTune · {sample_count:,} train / {stats['eval']:,} eval items · "
            f"{stats['hours']:.2f} train hours · mean {stats['mean_seconds']:.1f}s, "
            f"p95 {stats['p95_seconds']:.1f}s audio · p95 {stats['p95_text_chars']:.0f} text chars. "
            f"Profile: {epochs} epoch(s), batch {batch_size}, accumulation {grad_accum}, "
            f"{optimizer_steps_per_epoch} optimizer step(s)/epoch, {total_optimizer_steps} total optimizer step(s), "
            f"LR {learning_rate:.2e} with warmup+cosine (floor 20%), WD {weight_decay:g}, clip {max_grad_norm:g}, "
            f"save/eval cadence {epoch_cadence}/{epoch_cadence} epoch(s), LoRA r/a/dropout {rank}/{alpha}/{dropout:g} "
            f"on {preset}. Attention uses the supported fallback chain."
        ),
    }


def _cuda_available():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _training_model_path(base_choice):
    info = _catalog_info(base_choice)
    if info.get("mode") != "Base / Voice Clone":
        raise ValueError("Qwen-compatible fine-tuning requires a 12 Hz Base model.")
    if int(info.get("hz", 12)) != 12:
        raise ValueError("The Qwen-compatible fine-tuning path currently supports Qwen3-TTS 12 Hz Base.")
    model_path = _model_path_for(base_choice)
    if isinstance(model_path, Path) and model_path.is_dir():
        return str(model_path)
    # Training checkpoints are copied into the run directory after every
    # completed epoch. Resolve catalog IDs into the project's local model
    # library before starting the worker; passing a Hub ID here makes the
    # worker download successfully and then fail later at shutil.copytree().
    info = _catalog_info(base_choice)
    target = MODELS / _safe(info["id"].replace("/", "--"))
    target.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=info["id"], local_dir=str(target), local_dir_use_symlinks=False)
    except Exception as exc:
        raise RuntimeError(f"Could not materialize the selected base model locally: {exc}") from exc
    if not (target / "config.json").is_file():
        raise RuntimeError(f"The downloaded base model is incomplete: {target}")
    return str(target)


def _capture_training_output(process, log_path: Path, project: str, total_epochs: int):
    global _TRAIN_STATE
    with log_path.open("a", encoding="utf-8", buffering=1) as handle:
        for raw in iter(process.stdout.readline, ""):
            if not raw:
                break
            line = raw.rstrip()
            handle.write(raw)
            handle.flush()
            log(f"[TRAIN] {line}")
            match = re.search(r"Epoch\s+(\d+)\s*\|\s*Step\s*(\d+)\s*\|\s*Loss\s*:?\s*([0-9.eE+-]+|nan|inf|[-+]?inf)", line, re.IGNORECASE)
            if match:
                _TRAIN_STATE["epoch"] = int(match.group(1))
                _TRAIN_STATE["step"] = int(match.group(2))
                try:
                    _TRAIN_STATE["loss"] = float(match.group(3))
                except ValueError:
                    _TRAIN_STATE["loss"] = None
                _TRAIN_STATE["total_epochs"] = total_epochs
                _TRAIN_STATE["status"] = "Training"
                if not math.isfinite(_TRAIN_STATE["loss"] or 0.0):
                    _TRAIN_STATE["status"] = "Non-finite loss detected; worker stopping safely"
                    _TRAIN_STATE["nonfinite"] = True
            eval_stage_match = re.search(r"\[EVAL\]\s+step=(\d+)\s+stage=(.+)$", line)
            if eval_stage_match:
                _TRAIN_STATE["eval_step"] = int(eval_stage_match.group(1))
                _TRAIN_STATE["status"] = f"Evaluation · {eval_stage_match.group(2).strip()}"
            if "starting dedicated evaluation thread" in line:
                _TRAIN_STATE["status"] = "Evaluation · training paused"
            elif "dedicated evaluation thread still running" in line:
                _TRAIN_STATE["status"] = "Evaluation · still running; training paused"
            elif "dedicated evaluation thread completed; resuming training" in line:
                _TRAIN_STATE["status"] = "Training · evaluation complete; resumed"
            elif "evaluation thread ended with an error; training will resume" in line:
                _TRAIN_STATE["status"] = "Training · evaluation skipped; resumed"
            eval_match = re.search(
                r"\[EVAL\]\s+(?:epoch=(\d+)\s+\|\s+)?step=(\d+)\s+\|\s+generated=([0-9.eE+-]+)s\s+\|\s+elapsed=([0-9.eE+-]+)s",
                line,
            )
            if eval_match:
                if eval_match.group(1) is not None:
                    _TRAIN_STATE["eval_epoch"] = int(eval_match.group(1))
                _TRAIN_STATE["eval_step"] = int(eval_match.group(2))
                _TRAIN_STATE["eval_seconds"] = float(eval_match.group(3))
                _TRAIN_STATE["eval_elapsed"] = float(eval_match.group(4))
            schedule_match = re.search(
                r"\[TRAIN\]\s+Schedule\s+\|\s+steps_per_epoch=(\d+)\s+\|\s+total_steps=(\d+)\s+\|\s+epochs=(\d+)",
                line,
            )
            if schedule_match:
                _TRAIN_STATE["steps_per_epoch"] = int(schedule_match.group(1))
                _TRAIN_STATE["total_steps"] = int(schedule_match.group(2))
                _TRAIN_STATE["total_epochs"] = int(schedule_match.group(3))


def _finish_training(project, output_dir, mode, returncode, speaker_name="speaker_1", model_export_mode=DEFAULT_MODEL_EXPORT_MODE, publish_icl_variant=None, base_choice=None):
    global _TRAIN_STATE
    if publish_icl_variant is not None and model_export_mode == DEFAULT_MODEL_EXPORT_MODE:
        model_export_mode = normalize_model_export_mode(None, publish_icl_variant)
    else:
        model_export_mode = normalize_model_export_mode(model_export_mode)
    if returncode == 0:
        output = Path(output_dir)
        if "Full SFT" in str(mode) or str(mode).startswith("Official Qwen SFT") or str(mode).startswith("Official full SFT"):
            # Full SFT is now intentionally one hybrid artifact.  It remains
            # Base-compatible, so it supports transcript+audio ICL and the
            # speaker-only/no-transcript path from the same checkpoint.
            model_export_mode = "ICL-variant"
            checkpoints = sorted(output.glob("checkpoint-epoch-*"), key=lambda p: int(re.search(r"(\d+)$", p.name).group(1)) if re.search(r"(\d+)$", p.name) else -1)
            if checkpoints:
                publish_custom = False
                publish_icl = True
                ready = TRAINING / _safe(project) / "ready"
                ready_icl = TRAINING / _safe(project) / "ready_icl"
                if ready.exists():
                    shutil.rmtree(ready, ignore_errors=True)
                if ready_icl.exists():
                    shutil.rmtree(ready_icl, ignore_errors=True)
                icl_source = output / "icl_variant"
                # New workers save the hybrid Base checkpoint directly under
                # the final checkpoint.  Keep the old ``icl_variant`` layout
                # as a compatibility fallback for completed runs.
                if not icl_source.is_dir():
                    candidate = checkpoints[-1]
                    try:
                        candidate_config = json.loads((candidate / "config.json").read_text(encoding="utf-8"))
                    except Exception:
                        candidate_config = {}
                    candidate_metadata = {}
                    metadata_path = candidate / "qwen_easy_training.json"
                    if metadata_path.is_file():
                        try:
                            candidate_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                        except Exception:
                            candidate_metadata = {}
                    if candidate_config.get("tts_model_type") == "base" or candidate_metadata.get("variant") == "base_icl":
                        icl_source = candidate
                published_variants = []
                if publish_custom:
                    shutil.copytree(checkpoints[-1], ready)
                    (ready / "qwen_easy_training.json").write_text(json.dumps({
                        "project": project,
                        "speaker_name": _safe(speaker_name),
                        "training_type": "qwen_compatible_full_sft",
                        "variant": "custom_voice",
                        "supports_icl": False,
                        "model_export_mode": model_export_mode,
                        "icl_variant_published": bool(publish_icl),
                        "base_model_choice": str(base_choice or ""),
                        "model_size": _model_size_from_text(base_choice),
                        "source_checkpoint": str(checkpoints[-1]),
                    }, indent=2), encoding="utf-8")
                    published_variants.append(str(ready))
                icl_available = icl_source.is_dir() and (icl_source / "config.json").is_file() and (icl_source / "model.safetensors").is_file()
                if publish_icl and icl_available:
                    shutil.copytree(icl_source, ready_icl)
                    icl_metadata_path = ready_icl / "qwen_easy_training.json"
                    try:
                        icl_metadata = json.loads(icl_metadata_path.read_text(encoding="utf-8")) if icl_metadata_path.is_file() else {}
                    except Exception:
                        icl_metadata = {}
                    icl_metadata.update({
                        "project": project,
                        "speaker_name": _safe(speaker_name),
                        "training_type": "qwen_compatible_full_sft_icl_variant",
                        "variant": "base_icl",
                        "supports_icl": True,
                        "supports_speaker_only": True,
                        "model_export_mode": model_export_mode,
                        "icl_variant_published": True,
                        "base_model_choice": str(base_choice or icl_metadata.get("base_model_choice", "")),
                        "model_size": _model_size_from_text(
                            base_choice,
                            icl_metadata.get("model_size"),
                            icl_metadata.get("base_model"),
                        ),
                        "source_checkpoint": str(checkpoints[-1]),
                    })
                    icl_metadata_path.write_text(json.dumps(icl_metadata, indent=2), encoding="utf-8")
                    published_variants.append(str(ready_icl))
                if publish_icl and not icl_available:
                    _TRAIN_STATE["status"] = "Complete, but the hybrid ICL checkpoint was not produced."
                elif published_variants:
                    _TRAIN_STATE["status"] = "Complete · hybrid ICL checkpoint published: " + " and ".join(published_variants)
                else:
                    _TRAIN_STATE["status"] = "Complete, but no hybrid ICL checkpoint was published."
            else:
                _TRAIN_STATE["status"] = "Complete, but no checkpoint directory was found."
        else:
            adapter = Path(output_dir) / "adapter"
            destination = TRAINING / _safe(project) / "adapter"
            if destination.exists():
                shutil.rmtree(destination, ignore_errors=True)
            if adapter.is_dir():
                shutil.copytree(adapter, destination)
                metadata_path = destination / "qwen_easy_lora.json"
                metadata = {}
                if metadata_path.is_file():
                    try:
                        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    except Exception:
                        metadata = {}
                metadata.update({
                    "project": project,
                    "adapter_name": _safe(project),
                    "export_mode": PEFT_ADAPTER_EXPORT_MODE,
                    "base_conditioning": "ICL-compatible",
                    "supports_icl": True,
                })
                metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            _TRAIN_STATE["status"] = f"Complete · LoRA adapter published at {destination}"
        if not _TRAIN_STATE.get("completion_chimed"):
            play_done_chime()
            _TRAIN_STATE["completion_chimed"] = True
    else:
        _TRAIN_STATE["status"] = (
            "Stopped safely after a non-finite loss/gradient; lower Learning Rate or use Auto/SDPA, "
            "then resume from a valid checkpoint."
            if _TRAIN_STATE.get("nonfinite")
            else f"Failed (exit {returncode})"
        )
    # Freeze the wall-clock duration before clearing the active flags.  The UI
    # polls asynchronously and must not continue measuring from ``started``
    # after the worker has already finished.
    started_at = _TRAIN_STATE.get("started")
    if started_at is not None:
        _TRAIN_STATE["finished_elapsed"] = max(0.0, time.time() - float(started_at))
    _TRAIN_STATE["running"] = False
    _TRAIN_STATE["starting"] = False
    _TRAIN_STATE["returncode"] = returncode


def start_training(project, dataset_name, base_choice, training_mode, speaker_name, epochs, batch_size, grad_accum, learning_rate, save_every_epochs, rank, alpha, dropout, weight_decay, max_grad_norm, training_seed, attention, lr_schedule="warmup_cosine", warmup_ratio=0.05, min_lr_ratio=0.20, resume=None, eval_enabled=False, eval_text="", eval_every_epochs=1, eval_fast=False, eval_icl=False, model_export_mode=DEFAULT_MODEL_EXPORT_MODE, publish_icl_variant=None):
    global _TRAIN_PROC, _TRAIN_THREAD, _TRAIN_STATE
    if publish_icl_variant is not None and model_export_mode == DEFAULT_MODEL_EXPORT_MODE:
        model_export_mode = normalize_model_export_mode(None, publish_icl_variant)
    else:
        model_export_mode = normalize_model_export_mode(model_export_mode)
    is_lora_training = training_mode == "PEFT LoRA adapter (experimental)"
    if is_lora_training:
        model_export_mode = PEFT_ADAPTER_EXPORT_MODE
    else:
        # There is no separate non-ICL full-SFT deployment anymore.  The
        # trained Base-compatible checkpoint is the hybrid artifact; users
        # choose ICL or speaker-only conditioning in Inference.
        model_export_mode = "ICL-variant"
    # Evaluation is always prompted ICL for both supported training paths.
    # This keeps PEFT and Full SFT comparisons on the same conditioning route
    # and avoids the old CustomVoice-only evaluation mismatch.
    eval_icl = bool(eval_enabled)
    with _TRAIN_LOCK:
        if _TRAIN_PROC is not None and _TRAIN_PROC.poll() is None:
            return "A training run is already active."
        if not project or project == NONE or not dataset_name:
            return "Select a Training Project and prepared dataset first."
        root, _train_raw, eval_raw, train_codes, eval_codes = _dataset_paths(dataset_name)
        if not train_codes.is_file():
            return "Extract Qwen audio codes in Dataset Preparation before starting training."
        with _INFERENCE_LOCK:
            if _INFERENCE_ACTIVE_JOBS > 0:
                return "Stop the active inference job before starting a new training run."
        # The training worker is a separate process.  Release every GUI-side
        # Qwen/Faster-Qwen/Whisper object first so its CUDA allocations and
        # prompt caches cannot compete with the new training process.
        unload_all_models(reason=f"starting training for {project}", reset_compiler=True, trim_ram=True)
        try:
            base_path = _training_model_path(base_choice)
        except Exception as exc:
            return str(exc)
        resume_value = str(resume or FRESH_RESUME)
        resume_path = None
        if resume_value not in {FRESH_RESUME, NONE, ""}:
            candidate = Path(resume_value).expanduser()
            if not candidate.is_dir():
                return f"Resume checkpoint not found: {candidate}"
            resume_path = candidate.resolve()
            if training_mode == "PEFT LoRA adapter (experimental)":
                if not (resume_path / "adapter_config.json").is_file():
                    return f"Selected resume path is not a LoRA checkpoint: {resume_path}"
            elif not ((resume_path / "config.json").is_file() and (list(resume_path.glob("*.safetensors")) or list(resume_path.glob("*.bin")) or (resume_path / "model.safetensors.index.json").is_file())):
                return f"Selected resume path is not a complete Qwen checkpoint: {resume_path}"
        fresh_training = resume_value in {FRESH_RESUME, NONE, ""}
        if fresh_training:
            try:
                _reset_training_project(project)
            except Exception as exc:
                return f"Could not reset the Training Project for a fresh run: {exc}"
        run_dir = TRAINING / _safe(project) / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        script = FINETUNING / ("sft_lora_12hz.py" if training_mode == "PEFT LoRA adapter (experimental)" else "sft_12hz.py")
        trained_speaker = _safe(speaker_name or "speaker_1")
        evaluation_jsonl = eval_codes if eval_codes.is_file() and eval_codes.read_text(encoding="utf-8").strip() else (eval_raw if eval_raw.is_file() else train_codes)
        attention_arg = "auto" if attention.startswith("Auto") else ("flash_attention_2" if attention.startswith("Flash") else "sdpa" if attention.startswith("SDPA") else "eager")
        requested_lr = max(1e-8, float(learning_rate))
        safe_lr_limit = 1e-6 if training_mode == "PEFT LoRA adapter (experimental)" else 2e-6
        effective_lr = min(requested_lr, safe_lr_limit)
        lr_schedule = str(lr_schedule or "warmup_cosine").strip().lower()
        if lr_schedule not in {"constant", "warmup_cosine"}:
            lr_schedule = "warmup_cosine"
        warmup_ratio = max(0.0, min(0.25, float(warmup_ratio)))
        min_lr_ratio = max(0.01, min(1.0, float(min_lr_ratio)))
        init_model_path = base_path
        args = [sys.executable, str(script), "--init_model_path", init_model_path, "--output_model_path", str(run_dir), "--train_jsonl", str(train_codes), "--batch_size", str(max(1, int(batch_size))), "--lr", str(effective_lr), "--lr_schedule", lr_schedule, "--warmup_ratio", str(warmup_ratio), "--min_lr_ratio", str(min_lr_ratio), "--num_epochs", str(max(1, int(epochs))), "--speaker_name", trained_speaker, "--attention", attention_arg, "--gradient_accumulation_steps", str(max(1, int(grad_accum))), "--save_every_epochs", str(max(0, int(save_every_epochs))), "--weight_decay", str(max(0.0, float(weight_decay))), "--max_grad_norm", str(max(0.1, float(max_grad_norm))), "--seed", str(int(training_seed)), "--eval_jsonl", str(evaluation_jsonl), "--eval_every_epochs", str(max(0, int(eval_every_epochs))), "--eval_text", str(eval_text or "This is a fixed evaluation sample generated during training.")]
        if eval_enabled:
            args += ["--eval_enabled"]
        if eval_enabled and eval_fast:
            args += ["--eval_fast"]
        if eval_enabled:
            args += ["--eval_icl"]
        if not is_lora_training:
            args += ["--model_export_mode", model_export_mode]
        if training_mode == "PEFT LoRA adapter (experimental)":
            args += ["--rank", str(max(1, int(rank))), "--alpha", str(max(1, int(alpha))), "--dropout", str(float(dropout))]
            if resume_path is not None:
                args += ["--resume_adapter_path", str(resume_path)]
        elif resume_path is not None:
            args += ["--resume_model_path", str(resume_path)]
        log_path = run_dir / "training.log"
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        resume_note = f" · resume={resume_path.name}" if resume_path is not None else " · fresh"
        lr_note = f" · lr={effective_lr:.2e} · schedule={lr_schedule} · floor={min_lr_ratio:.0%}" + (f" (clamped from {requested_lr:.2e})" if effective_lr < requested_lr else "")
        export_note = "PEFT adapter (Base ICL compatible)" if is_lora_training else "hybrid ICL export"
        _TRAIN_STATE = {"running": True, "starting": True, "status": f"Preparing training worker{resume_note}{lr_note} · {export_note} · evaluation=Base ICL · models unloaded", "project": project, "started": time.time(), "finished_elapsed": None, "returncode": None, "log": str(log_path), "output_dir": str(run_dir), "epoch": 0, "total_epochs": int(epochs), "step": 0, "steps_per_epoch": 0, "loss": None, "eval_epoch": 0, "eval_step": 0, "eval_seconds": None, "eval_elapsed": None, "nonfinite": False, "eval_enabled": bool(eval_enabled), "eval_fast": bool(eval_fast), "eval_icl": bool(eval_icl), "eval_every_epochs": int(max(0, eval_every_epochs)), "save_every_epochs": int(max(0, save_every_epochs)), "resume": str(resume_path or FRESH_RESUME), "effective_lr": effective_lr, "lr_schedule": lr_schedule, "warmup_ratio": warmup_ratio, "min_lr_ratio": min_lr_ratio, "model_export_mode": model_export_mode, "completion_chimed": False}
        try:
            _TRAIN_PROC = subprocess.Popen(args, cwd=str(FINETUNING), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        except Exception as exc:
            _TRAIN_STATE.update({"running": False, "starting": False, "returncode": -1, "finished_elapsed": max(0.0, time.time() - float(_TRAIN_STATE.get("started") or time.time())), "status": f"Failed to start: {exc}"})
            return _TRAIN_STATE["status"]
        _TRAIN_STATE["starting"] = False

        def worker():
            global _TRAIN_PROC
            _capture_training_output(_TRAIN_PROC, log_path, project, int(epochs))
            code = _TRAIN_PROC.wait()
            _finish_training(project, run_dir, training_mode, code, trained_speaker, model_export_mode=model_export_mode, base_choice=base_choice)

        _TRAIN_THREAD = threading.Thread(target=worker, daemon=True, name="qwen-training-worker")
        _TRAIN_THREAD.start()
        return f"Training started for '{project}' using {training_mode}."


def stop_training():
    global _TRAIN_PROC
    with _TRAIN_LOCK:
        if _TRAIN_PROC is None or _TRAIN_PROC.poll() is not None:
            return "No active training run."
        _TRAIN_PROC.terminate()
        _TRAIN_STATE["status"] = "Stop requested"
        return "Stop requested; the training worker is being terminated."


def training_status():
    return dict(_TRAIN_STATE)


def training_progress_snapshot():
    state = dict(_TRAIN_STATE)
    total = max(1, int(state.get("total_epochs") or 1))
    epoch = max(0, int(state.get("epoch") or 0))
    step = max(0, int(state.get("step") or 0))
    steps_per_epoch = max(0, int(state.get("steps_per_epoch") or 0))
    total_steps = max(0, int(state.get("total_steps") or 0))
    if steps_per_epoch and total_steps:
        # The worker reports the current one-based epoch and global optimizer
        # step.  Convert that into fractional epoch progress so ETA appears
        # during the first epoch instead of waiting for its completion.
        completed_steps = min(total_steps, max(0, step))
        pct = min(100.0, max(0.0, 100.0 * completed_steps / total_steps))
    else:
        pct = min(100.0, max(0.0, 100.0 * float(epoch) / total))
    if state.get("running") or state.get("starting"):
        elapsed = time.time() - state["started"] if state.get("started") else None
    else:
        elapsed = state.get("finished_elapsed")
    return {**state, "pct": pct, "elapsed": elapsed, "eta": (elapsed * (100 - pct) / pct) if elapsed and pct > 0 else None}


def tensorboard_logdir(project):
    runs = sorted((TRAINING / _safe(project) / "runs").glob("*") if (TRAINING / _safe(project) / "runs").is_dir() else [], key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def clear_outputs():
    removed = 0
    for item in OUTPUTS.glob("*"):
        try:
            shutil.rmtree(item) if item.is_dir() else item.unlink()
            removed += 1
        except Exception:
            pass
    return f"Removed {removed} output item(s)."
