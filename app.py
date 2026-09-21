from __future__ import annotations

import html
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

import gradio as gr

import qwen_backend as B
from qwen_easy.console import html_view, log
from qwen_easy.projects import clone_project, delete_surface, list_projects, load_project, save_project, update_surface

APP_TITLE = "Qwen TTS Easy GUI"
NONE = B.NONE
MAX_DIALOGUE = 12
WORKFLOW_CHOICES = [
    ("CustomVoice — Qwen bundled speakers", "CustomVoice"),
    ("VoiceDesign — written voice description", "VoiceDesign"),
    ("Base / Voice Clone — reference audio", "Base / Voice Clone"),
    ("Trained Custom Voice — trained speaker", "Trained Custom Voice"),
]
INSTRUCTION_WORKFLOW_MODES = {"CustomVoice", "VoiceDesign", "Trained Custom Voice"}

_TENSORBOARD_LOCK = threading.Lock()
_TENSORBOARD_PROCESS = None
_TENSORBOARD_PORT = None
_TENSORBOARD_LOGDIR = None

CSS = """
html,body,#root{width:100%;min-width:0}
.gradio-container{max-width:none!important;width:100%!important;margin:0 auto!important;padding:18px 28px 28px!important}
.tabs,.tabitem,.gradio-container .block,.gradio-container .form{width:100%!important;max-width:none!important}
.title-section{border-bottom:1px solid var(--border-color-primary);margin-bottom:8px;padding-bottom:8px;align-items:center!important}
.title-section h1,.title-section .prose{margin:0!important;padding:0!important}.title-section button{min-height:36px!important;white-space:nowrap}
.tab-subtitle{opacity:.78;margin:0 0 12px!important}.section-heading{margin:10px 0 2px!important}
.toolbar,.runtime-row,.workflow-actions,.dialogue-toolbar,.project-strip{gap:10px!important;align-items:end!important}
.compact{min-width:52px!important;max-width:130px!important}.seed-layout{gap:14px!important;align-items:stretch!important}.seed-layout>.form{min-width:0!important;flex:1 1 0!important}
.seed-layout>.seed-action{flex:0 0 160px!important;width:160px!important;min-width:160px!important;max-width:160px!important;align-self:center!important}
.seed-action,.seed-action button{min-width:0!important;max-width:100%!important;min-height:74px!important;height:74px!important;white-space:nowrap!important}
.seed-action{display:flex!important;align-items:center!important}.seed-action button{width:100%!important;padding:0 12px!important}
.medium-control{max-width:420px!important}.small-control{max-width:260px!important}
.card,.dialogue-turn-card{padding:14px 14px 12px!important;margin:6px 0 12px!important;border:1px solid var(--border-color-primary)!important;border-radius:10px!important;background:transparent!important;box-shadow:none!important}
.dialogue-actions{gap:7px!important;align-items:center!important}.dialogue-actions button{min-width:88px!important;padding-left:10px!important;padding-right:10px!important;white-space:nowrap!important}
.audio-safe-space{overflow:visible!important;padding-bottom:18px!important;border:0!important;box-shadow:none!important}
.output-clean,.output-clean>div,.output-clean .wrap{border:0!important;box-shadow:none!important}
.compact-status{margin:0 0 8px!important;padding:0!important;min-height:0!important}
.status-card,.training-progress-card{padding:10px 12px;border:1px solid var(--border-color-primary);border-radius:9px;background:transparent}
.training-progress-head{display:flex;justify-content:space-between;gap:16px;margin-bottom:8px}
.training-progress-track{width:100%;height:10px;border-radius:999px;background:var(--background-fill-secondary);overflow:hidden}
.training-progress-fill{height:100%;border-radius:999px;background:var(--button-primary-background-fill);transition:width .25s ease}.training-progress-meta{margin-top:8px;opacity:.82;font-size:.92em}
.voice-library-row{align-items:end!important;gap:8px!important}.voice-library-row button{min-width:48px!important;max-width:52px!important}
.console-accordion,.console-accordion>div{border-radius:8px!important}footer{display:none!important}
@media(max-width:900px){.gradio-container{padding:10px!important}.title-section button{min-width:100%!important}.medium-control,.small-control{max-width:none!important}.seed-layout>.seed-action{flex:1 1 0!important;width:auto!important;min-width:0!important;max-width:none!important}}
"""


def voice_library_choices(mode=None):
    """Display Voice Library kind markers while keeping stable raw values."""
    return [(NONE, NONE), *B.get_sample_dropdown_choices(mode)]


def voices(mode=None):
    """Return raw Voice Library values for backend/state compatibility."""
    return [value for _label, value in voice_library_choices(mode)]


def models():
    return B.list_ready_models()


def default_model_choice():
    choices = models()
    for _label, value in choices:
        if "12Hz" in str(value) and "Base / Voice Clone" in str(value):
            return value
    return choices[0][1] if choices else None


def workflow_for_model(choice):
    try:
        return B._catalog_info(choice).get("mode", "Base / Voice Clone")
    except Exception:
        return "Base / Voice Clone"


def model_capability_details(choice):
    """Explain the conditioning surface exposed by the selected artifact."""
    try:
        info = B._catalog_info(choice)
    except Exception:
        return "Select a compatible Qwen checkpoint to see its supported conditioning controls."
    mode = str(info.get("mode", ""))
    if mode == "Base / Voice Clone":
        if info.get("variant") == "base_icl":
            packaged_speaker = B.has_packaged_speaker_prompt(choice)
            packaged_icl = B.has_packaged_icl_prompt(choice)
            packaged_text = ""
            if packaged_speaker and packaged_icl:
                packaged_text = " It bundles both speaker-only and ICL prompts, so when no external audio/transcript pair is supplied the existing Speaker-only boolean selects between those two internal paths."
            elif packaged_speaker:
                packaged_text = " Its learned speaker prompt is bundled, so reference-free Speaker-only generation is available; an internal ICL prompt is not present."
            else:
                packaged_text = " Reference-free generation is unavailable until a trained prompt is bundled."
            return "⚠️ **Unofficial Hybrid Trained variant** — one complete Base-compatible checkpoint with external ICL and Speaker-only conditioning." + packaged_text + " When external reference audio and its exact transcript are supplied, the current boolean keeps its existing behavior. This hybrid export is unofficial and may behave differently from an official Qwen checkpoint; LoRA is not applicable to this complete artifact."
        if B.has_packaged_speaker_prompt(choice):
            packaged_icl = B.has_packaged_icl_prompt(choice)
            internal = " A bundled ICL prompt is also used automatically only when no external audio/transcript pair is supplied." if packaged_icl else ""
            return "**Base / Voice Clone with packaged trained speaker** — ICL remains available with reference audio and its exact transcript, while Speaker-only mode can use the bundled speaker prompt without reference audio." + internal
        return "**Base / Voice Clone** — supports single or multiple reference audio, exact reference transcripts for ICL, and Speaker-only conditioning."
    if mode == "Trained Custom Voice":
        return "**Trained Custom Voice** — legacy complete Qwen CustomVoice checkpoint using the trained speaker. It is speaker-only and does not expose Base reference conditioning. Current Full SFT publishes the hybrid Base-compatible artifact as **Unofficial Hybrid Trained**."
    if mode == "CustomVoice":
        return "**CustomVoice** — uses Qwen's bundled speakers and optional instructions. It does not use reference audio or ICL."
    if mode == "VoiceDesign":
        return "**VoiceDesign** — creates the voice from a written voice description. It does not use reference audio or ICL."
    return "The selected checkpoint determines which conditioning controls are available."


def default_training_base_choice():
    for label in B.MODEL_CATALOG:
        if "1.7B" in label and "Base / Voice Clone" in label:
            return label
    return next(iter(B.MODEL_CATALOG), None)


def loras(base_choice=None):
    return B.list_lora_choices(base_choice)


def refresh_models(current=None):
    choices = models()
    values = [value for _label, value in choices]
    return gr.update(choices=choices, value=current if current in values else values[0])


def refresh_loras(base_choice=None, current=NONE):
    choices = loras(base_choice)
    values = [value for _label, value in choices]
    return gr.update(choices=choices, value=current if current in values else NONE)


def trained_checkpoint_update(model_choice, current=None):
    choices = B.list_trained_checkpoint_choices(model_choice)
    values = [value for _label, value in choices]
    selected = current if current in values else (values[0] if values else None)
    return gr.update(choices=choices, value=selected, visible=len(choices) > 1), selected or model_choice


def trained_checkpoint_selected(checkpoint, model_choice):
    return checkpoint or model_choice


def merge_lora_ui(base_model, adapter, scale, attention):
    status, selected = B.merge_lora_for_fast_inference(base_model, adapter, scale, attention)
    choices = models()
    values = [value for _label, value in choices]
    model_value = selected if selected in values else base_model
    adapter_choices = loras(base_model)
    adapter_value = NONE if selected else adapter
    return gr.update(choices=choices, value=model_value), gr.update(choices=adapter_choices, value=adapter_value), status


def refresh_voices(current=NONE, mode=None):
    choices = voice_library_choices(mode)
    values = [value for _label, value in choices]
    return gr.update(choices=choices, value=current if current in values else NONE)


def dialogue_voice_choices(mode, model_choice=None, reference_mode="Single reference mode", adapter=None):
    if mode == "CustomVoice":
        return [(speaker, speaker) for speaker in B.SPEAKERS]
    if mode == "Trained Custom Voice":
        return [(speaker, speaker) for speaker in (B.get_supported_speakers(model_choice) or ["speaker_1"])]
    if mode == "Base / Voice Clone":
        # Dialogue Builder deliberately uses explicit Voice Library entries
        # only.  Packaged trained prompts are a no-reference convenience for
        # Single Inference; making one the first row default caused every
        # dialogue turn to silently use the packaged speaker instead of the
        # selected saved reference.
        return voice_library_choices(reference_mode)
    return [(NONE, NONE)]


def dialogue_voice_label(mode):
    if mode == "CustomVoice":
        return "Bundled Speaker"
    if mode == "Trained Custom Voice":
        return "Trained Speaker"
    if mode == "Base / Voice Clone":
        return "Voice Library Reference"
    return "Speaker"


def speaker_field_details(mode):
    if mode == "CustomVoice":
        return "Bundled Speaker", "Qwen's bundled CustomVoice speaker list."
    if mode == "Trained Custom Voice":
        return "Trained Speaker", "Only the speaker exported by this trained checkpoint is shown."
    if mode == "Base / Voice Clone":
        return "Voice Library Reference", "Select a saved reference from Voice Library."
    return "Speaker", "This workflow does not use a speaker selector."


def instruction_field_details(mode):
    if mode == "VoiceDesign":
        return "Voice Description (required)", "Describe the voice naturally; Qwen uses this text to design the voice."
    if mode in INSTRUCTION_WORKFLOW_MODES:
        return "Instruction (optional)", "Optional style/prosody instruction sent to Qwen, or loaded from Saved Instruction Library."
    return "Instruction", "This workflow does not use an instruction field."


def refresh_dialogue_voices(mode, model_choice, reference_mode="Single reference mode", adapter=None):
    choices = dialogue_voice_choices(mode, model_choice, reference_mode, adapter)
    values = [value for _label, value in choices]
    fallback = values[0] if values else NONE
    label = dialogue_voice_label(mode)
    visible = mode != "VoiceDesign"
    return [gr.update(choices=choices, value=fallback, label=f"Turn {index + 1} · {label}", visible=visible) for index in range(MAX_DIALOGUE)]


def instruction_choices():
    return B.list_instruction_choices()


def load_instruction_ui(name):
    return B.load_instruction(name)


def refresh_instructions(current=NONE):
    choices = instruction_choices()
    values = [value for _label, value in choices]
    return gr.update(choices=choices, value=current if current in values else NONE)


def save_instruction_ui(name, text, current=NONE):
    status = B.save_instruction(name, text)
    choices = instruction_choices()
    values = [value for _label, value in choices]
    selected = name if name in values and "Saved instruction" in status else (current if current in values else NONE)
    return gr.update(choices=choices, value=selected), status


def delete_instruction_ui(name):
    status = B.delete_instruction(name)
    return refresh_instructions(), status


def library_visibility(mode):
    multiple = mode == "Multiple reference mode"
    return gr.update(visible=not multiple), gr.update(visible=multiple)


def load_library_voice(name, mode="Single reference mode"):
    audio, transcript, language, status = B.load_sample(name)
    multiple = mode == "Multiple reference mode"
    if multiple:
        return None, audio if isinstance(audio, list) else ([audio] if audio else None), transcript, language, status
    return (audio[0] if isinstance(audio, list) and audio else audio), None, transcript, language, status


def reference_library_mode(value):
    return "Multiple reference mode" if value == "Multiple audio references" else "Single reference mode"


def reference_mode_changed(mode, current=NONE):
    """Atomically switch the library choices and load/clear its conditioning.

    Gradio can dispatch multiple listeners for one Dropdown change close
    together. Updating the choices in one listener and loading the old value
    in another briefly leaves the request with a stale dropdown value (for
    example a single-reference entry while the choices already contain only
    multi-reference entries). Returning every affected component together
    prevents Dropdown preprocessing from rejecting that transient value.
    """
    sample_mode = reference_library_mode(mode)
    choices = voice_library_choices(sample_mode)
    values = [value for _label, value in choices]
    selected = current if current in values else NONE
    if selected == NONE:
        return gr.update(choices=choices, value=NONE), None, None, "", "Auto", "No saved voice selected."
    single_audio, multiple_audio, transcript, language, status = load_library_voice(selected, sample_mode)
    return gr.update(choices=choices, value=selected), single_audio, multiple_audio, transcript, language, status


def save_voice_ui(single_audio, multiple_audio, name, transcript, language, mode):
    selected = multiple_audio if mode == "Multiple reference mode" else single_audio
    status, saved = B.save_sample(selected, name, transcript, language, mode)
    choices = voice_library_choices(mode)
    values = [value for _label, value in choices]
    return status, gr.update(choices=choices, value=saved if saved in values else NONE)


def delete_voice(name, mode):
    status = B.delete_sample(name)
    return gr.update(choices=voice_library_choices(mode), value=NONE), None, None, "", "Auto", status


def transcribe_reference(audio, whisper):
    # The dropdown displays a VRAM-qualified label, while Faster-Whisper
    # expects the canonical model id (for example, ``large-v3``).
    text, status = B.transcribe_only(audio, B.whisper_model_id(whisper), "Auto")
    return text, status


def mode_visibility(mode):
    clone = mode == "Base / Voice Clone"
    speaker_mode = mode in {"CustomVoice", "Trained Custom Voice"}
    instruction_mode = mode in INSTRUCTION_WORKFLOW_MODES
    speaker_label, speaker_info = speaker_field_details(mode)
    instruction_label, instruction_info = instruction_field_details(mode)
    return gr.update(visible=speaker_mode, label=speaker_label, info=speaker_info), gr.update(visible=instruction_mode, label=instruction_label, info=instruction_info), gr.update(visible=clone), gr.update(visible=clone), gr.update(visible=clone), gr.update(visible=clone), gr.update(visible=instruction_mode)


def dialogue_mode_visibility(mode):
    instruction_mode = mode in INSTRUCTION_WORKFLOW_MODES
    instruction_label, instruction_info = instruction_field_details(mode)
    return gr.update(visible=mode == "Base / Voice Clone"), gr.update(visible=instruction_mode), gr.update(visible=instruction_mode, label=instruction_label, info=instruction_info)


def lora_visibility(mode, model_choice=None, current=NONE):
    """Show LoRA only where a separate adapter can actually be applied."""
    trained_icl = False
    try:
        trained_icl = B._catalog_info(model_choice).get("variant") == "base_icl"
    except Exception:
        pass
    visible = mode == "Base / Voice Clone" and not trained_icl
    if visible:
        choices = loras(model_choice)
        values = [value for _label, value in choices]
        return (
            gr.update(visible=True, choices=choices, value=current if current in values else NONE),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(
                visible=True,
                value="Base + LoRA: Standard Qwen can apply the adapter live. Faster Qwen CUDA Graphs requires a merged standalone checkpoint."
            ),
        )
    if trained_icl:
        status = "Unofficial Hybrid Trained is already a complete Base-compatible checkpoint; LoRA and merge are not applicable. Faster Qwen may load it directly when its 12 Hz/reference constraints are met."
    elif mode == "Trained Custom Voice":
        status = "Trained Custom Voice is already a complete checkpoint; no LoRA adapter or merge is needed."
    elif mode == "CustomVoice":
        status = "CustomVoice is a complete Qwen checkpoint; no LoRA adapter or merge is needed."
    else:
        status = "LoRA is available only for the Base / Voice Clone workflow."
    return (
        gr.update(visible=False, value=NONE),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False, value=status),
    )


def model_changed(choice):
    mode = workflow_for_model(choice)
    speakers = B.get_supported_speakers(choice) or B.SPEAKERS
    return gr.update(value=mode), gr.update(choices=speakers, value=speakers[0] if speakers else NONE), model_capability_details(choice)


def dialogue_workflow_update(choice):
    return gr.update(value=workflow_for_model(choice))


def resolve_seed(seed, fixed):
    return (B.normalize_seed(seed) or B.random_seed_value()) if fixed else B.random_seed_value()


def new_seed_pair():
    value = B.random_seed_value()
    return value, value


def run_inference(model, mode, text, language, speaker, instruct, voice_name, reference_mode, ref_audio, multiple_refs, ref_text, x_vector_only, adapter, lora_scale, temperature, top_p, top_k, repetition, max_tokens, chunk_mode, chunk_gap, attention, engine, seed, fixed, progress=gr.Progress(track_tqdm=True)):
    chosen = resolve_seed(seed, fixed)
    refs = multiple_refs if reference_mode == "Multiple audio references" else ref_audio
    audio, status, chosen = B.synthesize(model, mode, text, language, speaker, instruct, voice_name, refs, ref_text, x_vector_only, adapter, lora_scale, temperature, top_p, top_k, repetition, max_tokens, chunk_mode, chunk_gap, attention, chosen, progress=progress, engine=engine)
    return audio, status, chosen, chosen


def load_preview(name, mode):
    return load_library_voice(name, "Multiple reference mode" if mode == "Multiple audio references" else "Single reference mode")[-1], *load_library_voice(name, "Multiple reference mode" if mode == "Multiple audio references" else "Single reference mode")[:2]


def reference_visibility(mode):
    multiple = mode == "Multiple audio references"
    return gr.update(visible=not multiple), gr.update(visible=multiple)


def dialogue_mutate(action, index, count, *args):
    voices_values = list(args[:MAX_DIALOGUE])
    texts = list(args[MAX_DIALOGUE:2 * MAX_DIALOGUE])
    count = int(count)
    if action in {"insert", "clone"} and count < MAX_DIALOGUE:
        voices_values.insert(index + 1, voices_values[index])
        texts.insert(index + 1, texts[index] if action == "clone" else "")
        voices_values.pop(); texts.pop(); count += 1
    elif action == "remove" and count > 1:
        voices_values.pop(index); texts.pop(index); voices_values.append(NONE); texts.append(""); count -= 1
    return [count] + [gr.update(value=voices_values[i], visible=i < count) for i in range(MAX_DIALOGUE)] + [gr.update(value=texts[i], visible=i < count) for i in range(MAX_DIALOGUE)] + [gr.update(visible=i < count) for i in range(MAX_DIALOGUE)]


def run_dialogue(model, mode, language, speaker, instruct, adapter, lora_scale, temperature, top_p, top_k, repetition, max_tokens, chunk_mode, chunk_gap, silence, attention, engine, seed, fixed, count, *args, progress=gr.Progress(track_tqdm=True)):
    # ``*args`` must come before the keyword-only Gradio progress object.
    # Otherwise the first voice dropdown binds to ``progress`` and every
    # dialogue voice/text value shifts by one position.
    chosen = resolve_seed(seed, fixed)
    return (*B.generate_dialogue(model, mode, language, speaker, instruct, adapter, lora_scale, temperature, top_p, top_k, repetition, max_tokens, chunk_mode, chunk_gap, silence, chosen, fixed, count, *args, attention=attention, progress=progress, engine=engine), chosen)


def browse_folder(current=""):
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
        value = filedialog.askdirectory(initialdir=current or str(Path.cwd()))
        root.destroy()
        return value or current
    except Exception as exc:
        log(f"Folder picker unavailable: {exc}", level="WARN")
        return current


def clear_outputs():
    return B.clear_outputs()


def project_update(surface, current=NONE):
    choices = [NONE, *list_projects(surface)]
    return gr.update(choices=choices, value=current if current in choices else NONE)


def create_project_ui(name, surface):
    try:
        created = save_project(name, {surface: {}})
        return project_update(surface, created), f"Created project '{created}'."
    except Exception as exc:
        return project_update(surface), f"Create failed: {exc}"


def clone_project_ui(name, surface):
    try:
        cloned = clone_project(name, surface=surface)
        return project_update(surface, cloned), f"Cloned '{name}' → '{cloned}'."
    except Exception as exc:
        return project_update(surface, name), f"Clone failed: {exc}"


def delete_project_ui(name, surface):
    if not name or name == NONE:
        return project_update(surface), "Select a project first."
    status = delete_surface(name, surface)
    return project_update(surface), status


def save_dataset_project_ui(project, source, language, whisper):
    if not project or project == NONE:
        return project_update("dataset"), "Select a Dataset Project first."
    update_surface(project, "dataset", {"source_folder": source or "", "language": language, "whisper_model": B.whisper_model_id(whisper)})
    return project_update("dataset", project), f"Saved Dataset state for '{project}'."


def load_dataset_project(project):
    data = load_project(project).get("dataset", {}) if project and project != NONE else {}
    whisper = data.get("whisper_model", B.DEFAULT_WHISPER_MODEL)
    whisper_label = next((label for label, value in B.WHISPER_MODELS.items() if value == whisper), whisper)
    return data.get("source_folder", ""), data.get("language", "Auto"), whisper_label


def prepare_dataset_ui(source, project, language, whisper, batch):
    if not project or project == NONE:
        return "Select or create a Dataset Project first.", "", "", ""
    status, train_raw, eval_raw = B.prepare_dataset(source, project, language, whisper, batch, play_chime=False)
    if train_raw:
        # Keep the logical project shared by Dataset Preparation and
        # Fine-tuning, including projects entered through the custom-value
        # field instead of the Create Project button.
        update_surface(project, "dataset", {"source_folder": source or "", "language": language, "whisper_model": B.whisper_model_id(whisper)})
        code_status = B.extract_audio_codes(project, B.DEFAULT_TOKENIZER_MODEL, "auto", play_chime=True)
        status = f"{status}\n\n{code_status}"
    return status, train_raw, eval_raw, project_update("dataset", project)


def sync_dataset_completion(dataset_project_name, current_training_project=NONE, current_dataset=NONE):
    """Synchronize a completed dataset into the Fine-tuning surface.

    Dataset and training state remain separate, but a prepared dataset uses
    the same logical project name in both tabs. This mirrors the project-aware
    workflow used by the other Easy GUIs and also handles a project typed into
    the Dataset Project dropdown rather than created explicitly first.
    """
    dataset_choices = [NONE, *B.list_datasets()]
    candidate = B._safe(dataset_project_name) if dataset_project_name and dataset_project_name != NONE else NONE
    dataset_ready = candidate in dataset_choices
    dataset_value = candidate if dataset_ready else (current_dataset if current_dataset in dataset_choices else NONE)
    training_choices = [NONE, *list_projects("training")]
    training_value = candidate if dataset_ready and candidate in training_choices else (current_training_project if current_training_project in training_choices else NONE)
    return gr.update(choices=dataset_choices, value=dataset_value), gr.update(choices=training_choices, value=training_value)


def save_training_project_ui(project, dataset, base, mode, speaker, hardware, epochs, batch, grad, lr, lr_schedule, warmup_ratio, min_lr_ratio, save_every_epochs, eval_epochs, rank, alpha, dropout, weight_decay, max_grad_norm, training_seed, attention, resume, eval_enabled, eval_text, eval_icl=False, eval_fast=False, model_export_mode=B.DEFAULT_MODEL_EXPORT_MODE):
    if not project or project == NONE:
        return project_update("training"), "Select a Training Project first."
    export_mode = B.PEFT_ADAPTER_EXPORT_MODE if str(mode).startswith("PEFT") else "ICL-variant"
    update_surface(project, "training", {"dataset": dataset, "base": base, "mode": mode, "speaker": speaker, "hardware": hardware, "epochs": epochs, "batch_size": batch, "grad_accum": grad, "learning_rate": lr, "lr_schedule": str(lr_schedule or "warmup_cosine"), "warmup_ratio": float(warmup_ratio), "min_lr_ratio": float(min_lr_ratio), "save_every_epochs": int(save_every_epochs), "rank": rank, "alpha": alpha, "dropout": dropout, "weight_decay": weight_decay, "max_grad_norm": max_grad_norm, "training_seed": training_seed, "attention": attention, "resume": resume, "model_export_mode": export_mode, "eval": {"enabled": bool(eval_enabled), "text": str(eval_text or ""), "every_epochs": int(eval_epochs), "faster_qwen": bool(eval_fast), "icl": bool(eval_enabled)}})
    return project_update("training", project), f"Saved Training state for '{project}'."


def load_training_project(project):
    data = load_project(project).get("training", {}) if project and project != NONE else {}
    dataset_choices = B.list_datasets()
    dataset = data.get("dataset") if data.get("dataset") in dataset_choices else (dataset_choices[0] if dataset_choices else NONE)
    base = data.get("base", default_training_base_choice())
    evaluation = data.get("eval", {}) or {}
    eval_enabled = bool(evaluation.get("enabled", data.get("eval_enabled", False)))
    eval_text = str(evaluation.get("text", data.get("eval_text", "This is a fixed evaluation sample generated during training.")) or "")
    eval_icl = bool(eval_enabled)
    eval_fast = bool(evaluation.get("faster_qwen", data.get("eval_fast", False)))
    model_export_mode = B.PEFT_ADAPTER_EXPORT_MODE if str(data.get("mode", "")).startswith("PEFT") else "ICL-variant"
    eval_epochs = int(evaluation.get("every_epochs", data.get("eval_epochs", 1)) or 1)
    save_epochs = int(data.get("save_every_epochs", 0) or 0)
    mode = data.get("mode", "Qwen-compatible Full SFT")
    # Migrate labels used by older project files without carrying the old
    # experimental marker into the current UI.
    if str(mode).startswith("Official") or str(mode) == "Qwen-compatible Full SFT (experimental)":
        mode = "Qwen-compatible Full SFT"
    learning_rate = float(data.get("learning_rate", 2e-6) or 2e-6)
    learning_rate = min(learning_rate, 1e-6 if str(mode).startswith("PEFT") else 2e-6)
    lr_schedule = str(data.get("lr_schedule", "warmup_cosine") or "warmup_cosine")
    if lr_schedule not in {value for _label, value in B.TRAINING_LR_SCHEDULE_CHOICES}:
        lr_schedule = "warmup_cosine"
    warmup_ratio = float(data.get("warmup_ratio", 0.05) or 0.05)
    min_lr_ratio = float(data.get("min_lr_ratio", 0.20) or 0.20)
    resume_choices = B.training_resume_choices(project, mode)
    resume_values = {value for _label, value in resume_choices}
    resume = str(data.get("resume", B.FRESH_RESUME) or B.FRESH_RESUME)
    if resume not in resume_values:
        resume = B.FRESH_RESUME
    attention = data.get("attention", B.TRAINING_ATTENTION_CHOICES[0])
    if attention not in B.TRAINING_ATTENTION_CHOICES:
        attention = B.TRAINING_ATTENTION_CHOICES[0]
    return (gr.update(value=dataset), gr.update(value=base), gr.update(value=mode), data.get("speaker", "speaker_1"), data.get("hardware", "24 GB VRAM (1.7B minimum)"), data.get("epochs", 3), data.get("batch_size", 1), data.get("grad_accum", 4), learning_rate, gr.update(choices=B.TRAINING_LR_SCHEDULE_CHOICES, value=lr_schedule), warmup_ratio, min_lr_ratio, save_epochs, eval_epochs, data.get("rank", 16), data.get("alpha", 32), data.get("dropout", .05), data.get("weight_decay", .01), data.get("max_grad_norm", 1.0), data.get("training_seed", 1234), gr.update(value=attention), gr.update(choices=resume_choices, value=resume), eval_enabled, eval_text, training_eval_icl_visibility(mode, eval_icl), eval_fast, gr.update(visible=eval_enabled), model_export_update(mode, model_export_mode))


def start_training_ui(project, dataset, base, mode, speaker, hardware, epochs, batch, grad, lr, lr_schedule, warmup_ratio, min_lr_ratio, save_every_epochs, eval_epochs, rank, alpha, dropout, weight_decay, max_grad_norm, training_seed, attention, resume, eval_enabled, eval_text, eval_icl=False, eval_fast=False, model_export_mode=B.DEFAULT_MODEL_EXPORT_MODE):
    return B.start_training(project, dataset, base, mode, speaker, epochs, batch, grad, lr, save_every_epochs, rank, alpha, dropout, weight_decay, max_grad_norm, training_seed, attention, lr_schedule=lr_schedule, warmup_ratio=warmup_ratio, min_lr_ratio=min_lr_ratio, resume=resume, eval_enabled=eval_enabled, eval_text=eval_text, eval_every_epochs=eval_epochs, eval_fast=eval_fast, eval_icl=eval_icl, model_export_mode=model_export_mode)


def refresh_training_resume_ui(project, mode, current=B.FRESH_RESUME):
    choices = B.training_resume_choices(project, mode)
    values = {value for _label, value in choices}
    selected = current if current in values else (choices[0][1] if choices else B.FRESH_RESUME)
    return gr.update(choices=choices, value=selected)


def model_export_visibility(mode, value=B.DEFAULT_MODEL_EXPORT_MODE):
    full_sft = "Full SFT" in str(mode or "") and not str(mode or "").startswith("PEFT")
    info = "Full SFT always publishes one hybrid Base-compatible checkpoint. With an external reference pair, the existing boolean selects ICL or Speaker-only conditioning; new exports also bundle a canonical ICL prompt and speaker prompt for use when no external pair is supplied. No duplicate CustomVoice export is created."
    if full_sft:
        return gr.update(
            visible=False,
            interactive=False,
            choices=["ICL-variant"],
            value="ICL-variant",
            label="Model Export (fixed hybrid ICL)",
            info=info,
        )
    return gr.update(
        visible=False,
        interactive=False,
        choices=[B.PEFT_ADAPTER_EXPORT_MODE],
        value=B.PEFT_ADAPTER_EXPORT_MODE,
        label="PEFT adapter (Base ICL compatible)",
        info="PEFT publishes only an adapter. Standard Qwen loads it over the selected Base model, and the Evaluation Zone always uses Base ICL without merging a checkpoint.",
    )


def model_export_update(mode, model_export_mode=B.DEFAULT_MODEL_EXPORT_MODE):
    return model_export_visibility(mode, model_export_mode)


def training_eval_icl_visibility(mode, value=False):
    return gr.update(
        visible=False,
        interactive=False,
        value=True,
    )


def training_advanced_visibility(mode):
    """Expose LoRA/attention controls only for the PEFT training path."""
    return gr.update(visible=str(mode or "").startswith("PEFT"))


def autotune_training_ui(dataset, base, mode, hardware):
    try:
        config = B.autotune_training(dataset, base, mode, hardware)
        return (config["epochs"], config["batch_size"], config["grad_accum"], config["learning_rate"], gr.update(choices=B.TRAINING_LR_SCHEDULE_CHOICES, value=config["lr_schedule"]), config["warmup_ratio"], config["min_lr_ratio"], config["save_every_epochs"], config["eval_every_epochs"], config["rank"], config["alpha"], config["dropout"], config["weight_decay"], config["max_grad_norm"], config["training_seed"], config["attention"], config["summary"])
    except Exception as exc:
        return (3, 1, 4, 2e-6, gr.update(choices=B.TRAINING_LR_SCHEDULE_CHOICES, value="warmup_cosine"), .05, .20, 0, 1, 16, 32, .05, .01, 1.0, 1234, B.TRAINING_ATTENTION_CHOICES[0], f"AutoTune failed: {exc}")


def _format_eta(seconds, state):
    if state == "Complete":
        return "0s"
    if state != "Running" or seconds is None:
        return "--"
    try:
        remaining = max(0, int(round(float(seconds))))
    except (TypeError, ValueError):
        return "--"
    hours, remainder = divmod(remaining, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _format_elapsed(seconds):
    if seconds is None:
        return "--"
    try:
        elapsed = max(0, int(round(float(seconds))))
    except (TypeError, ValueError):
        return "--"
    hours, remainder = divmod(elapsed, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def training_progress_html(snapshot=None):
    snap = snapshot or B.training_progress_snapshot()
    pct = min(100.0, max(0.0, float(snap.get("pct", 0) or 0)))
    state = "Preparing" if snap.get("starting") else ("Running" if snap.get("running") else ("Complete" if snap.get("returncode") == 0 else ("Failed" if snap.get("returncode") is not None else "Idle")))
    eta = _format_eta(snap.get("eta"), state)
    elapsed = _format_elapsed(snap.get("elapsed", snap.get("finished_elapsed")))
    loss = "--" if snap.get("loss") is None else f"{float(snap['loss']):.5f}"
    eval_epoch = snap.get("eval_epoch") or 0
    eval_step = snap.get("eval_step") or 0
    eval_meta = f" · Eval epoch {eval_epoch} · step {eval_step}" if eval_step else ""
    meta = f"{html.escape(str(snap.get('status') or 'Waiting'))} · Epoch {snap.get('epoch', 0)}/{snap.get('total_epochs', '--')} · Step {snap.get('step', 0)} · Loss {loss}{eval_meta}"
    return f'<div class="training-progress-card"><div class="training-progress-head"><b>{state}</b><span>{pct:.1f}% · Elapsed {elapsed} · ETA {eta}</span></div><div class="training-progress-track"><div class="training-progress-fill" style="width:{pct:.2f}%"></div></div><div class="training-progress-meta">{meta}</div></div>'


def training_poll_ui():
    snap = B.training_progress_snapshot()
    running = bool(snap.get("running") or snap.get("starting"))
    return training_progress_html(snap), gr.update(interactive=not running, value="Preparing..." if snap.get("starting") else ("Training..." if running else "🚀 Start Training")), gr.update(interactive=running)


def inference_controls_idle():
    return (
        gr.update(value="Generate Audio 🚀", interactive=True),
        gr.update(interactive=False),
        gr.update(value="Generate Dialogue 🚀", interactive=True),
        gr.update(interactive=False),
    )


def inference_controls_running():
    return (
        gr.update(value="Generating Audio...", interactive=False),
        gr.update(value="🛑 Stop", interactive=True),
        gr.update(value="Generating Dialogue...", interactive=False),
        gr.update(value="🛑 Stop", interactive=True),
    )


def inference_poll_ui():
    if B.inference_status_snapshot().get("running"):
        return inference_controls_running()
    return inference_controls_idle()


def open_tensorboard(project):
    global _TENSORBOARD_PROCESS, _TENSORBOARD_PORT, _TENSORBOARD_LOGDIR
    logdir = B.tensorboard_logdir(project)
    if not logdir:
        return "No training run exists for this project yet."
    try:
        logdir = Path(logdir).resolve()

        def port_is_listening(port):
            try:
                with socket.create_connection(("127.0.0.1", int(port)), timeout=0.5):
                    return True
            except OSError:
                return False

        def free_port():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind(("127.0.0.1", 0))
                return int(probe.getsockname()[1])

        with _TENSORBOARD_LOCK:
            running = _TENSORBOARD_PROCESS is not None and _TENSORBOARD_PROCESS.poll() is None
            if running and _TENSORBOARD_LOGDIR == logdir and _TENSORBOARD_PORT and port_is_listening(_TENSORBOARD_PORT):
                port = int(_TENSORBOARD_PORT)
            else:
                # Keep one GUI-owned TensorBoard child at a time. This also
                # prevents opening a stale project when the user switches
                # projects and presses the button again.
                if running:
                    _TENSORBOARD_PROCESS.terminate()
                    try:
                        _TENSORBOARD_PROCESS.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        _TENSORBOARD_PROCESS.kill()
                _TENSORBOARD_PROCESS = None
                _TENSORBOARD_PORT = None
                _TENSORBOARD_LOGDIR = None

                process = None
                port = None
                for _attempt in range(4):
                    candidate = free_port()
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "tensorboard.main",
                            "--logdir",
                            str(logdir),
                            "--host",
                            "127.0.0.1",
                            "--port",
                            str(candidate),
                        ],
                        cwd=str(B.ROOT),
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    deadline = time.monotonic() + 15.0
                    while time.monotonic() < deadline:
                        if process.poll() is not None:
                            break
                        if port_is_listening(candidate):
                            port = candidate
                            break
                        time.sleep(0.2)
                    if port is not None:
                        break
                    if process.poll() is None:
                        process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()

                if port is None or process is None or process.poll() is not None:
                    return "TensorBoard could not become ready on a local port."
                _TENSORBOARD_PROCESS = process
                _TENSORBOARD_PORT = port
                _TENSORBOARD_LOGDIR = logdir

        url = f"http://127.0.0.1:{int(port)}"
        webbrowser.open_new(url)
        return f"TensorBoard opened for `{logdir.name}`: {url}"
    except Exception as exc:
        return f"TensorBoard could not start: {exc}"


with gr.Blocks(title=APP_TITLE, theme=gr.themes.Default(), css=CSS) as app:
    initial_model_choice = default_model_choice()
    initial_workflow = workflow_for_model(initial_model_choice)
    initial_checkpoint_choices = B.list_trained_checkpoint_choices(initial_model_choice)
    try:
        initial_lora_visible = initial_workflow == "Base / Voice Clone" and B._catalog_info(initial_model_choice).get("variant") != "base_icl"
    except Exception:
        initial_lora_visible = initial_workflow == "Base / Voice Clone"
    initial_speaker_label, initial_speaker_info = speaker_field_details(initial_workflow)
    initial_instruction_label, initial_instruction_info = instruction_field_details(initial_workflow)
    with gr.Row(elem_classes="title-section"):
        with gr.Column(scale=7):
            gr.Markdown("# 🎙️ Qwen TTS Easy GUI")
            gr.Markdown("Qwen3-TTS · reusable Voice Library · reference-aware inference · smart dataset preparation · complete CustomVoice SFT · PEFT/LoRA inference", elem_classes="tab-subtitle")
        with gr.Column(scale=3):
            with gr.Row():
                unload = gr.Button("🧹 Unload Models / Free VRAM", variant="secondary")
                clear = gr.Button("🗑️ Clear Outputs", variant="secondary")
    hidden = gr.Markdown(visible=False)
    unload.click(B.unload_all_models, outputs=hidden, queue=False)
    clear.click(clear_outputs, outputs=hidden, queue=False)

    with gr.Tabs():
        with gr.Tab("🎙️ Prep Samples"):
            gr.Markdown("*Create reusable Qwen speaker references. The same Voice Library is available in Single Inference and Dialogue Builder.*", elem_classes="tab-subtitle")
            with gr.Accordion("📖 Quick Guide", open=False):
                gr.Markdown("""1. Choose **Single reference mode** for one clip or **Multiple reference mode** for a reusable package of several clips. Upload files or record a reference from the microphone.
2. Give the entry a stable **Voice Name** and press **Save Voice**. A multiple-reference entry keeps all selected clips together and can be reused from Single Inference or Dialogue Builder.
3. **↻ Refresh** rescans the Voice Library after files are added or changed. **🗑️** removes the selected saved voice and its package files.
4. For **external ICL cloning**, enter the exact words spoken in each reference. When clips contain different words, keep a matching transcript for each one. Use **Speaker-only mode** when a transcript is unavailable. Trained hybrid checkpoints can use their own bundled prompt when no external reference pair is supplied.
5. **Faster-Whisper Transcription** can draft a transcript for the selected audio. Review and correct the result before saving. Voice Library audio is normalized to 24 kHz mono WAV.""")
            with gr.Row():
                with gr.Column():
                    voice_mode = gr.Dropdown(["Single reference mode", "Multiple reference mode"], value="Single reference mode", label="Voice Library Reference Mode", info="Single stores one clip. Multiple stores a reusable Qwen reference list.")
                    with gr.Row(elem_classes="voice-library-row"):
                        voice = gr.Dropdown(voice_library_choices("Single reference mode"), value=NONE, label="Voice Library", scale=8)
                        voice_refresh = gr.Button("↻", elem_classes="compact"); voice_delete = gr.Button("🗑️", variant="stop", elem_classes="compact")
                    single_audio = gr.Audio(type="filepath", sources=["upload", "microphone"], label="Reference Audio", elem_classes="audio-safe-space")
                    multiple_audio = gr.File(file_count="multiple", file_types=["audio"], type="filepath", label="Multiple Reference Audio (native multi-file)", visible=False)
                    voice_transcript = gr.Textbox(label="Exact Reference Transcript", lines=4, info="Required for external Qwen ICL cloning; leave blank only for speaker-only conditioning or when a trained hybrid provides its own bundled prompt.")
                    voice_language = gr.Dropdown(B.LANGUAGE_CHOICES, value="Auto", label="Reference Language")
                    with gr.Accordion("🛰️ Faster-Whisper Transcription", open=False):
                        whisper_ref = gr.Dropdown(list(B.WHISPER_MODELS), value=B.DEFAULT_WHISPER_MODEL, label="Whisper Model", info="Faster-Whisper model used only to draft the reference transcript.")
                        with gr.Row(elem_classes="workflow-actions"):
                            transcribe = gr.Button("Transcribe Reference", variant="secondary", scale=5)
                            transcribe_stop = gr.Button("🛑 Stop", variant="stop", scale=1)
                with gr.Column():
                    voice_name = gr.Textbox(label="Voice Name", placeholder="speaker_name")
                    save_voice = gr.Button("Save Voice", variant="primary")
                    voice_status = gr.Markdown("No saved voice selected.", elem_classes="compact-status")
            voice_mode.change(library_visibility, voice_mode, [single_audio, multiple_audio], queue=False)
            voice_mode.change(lambda mode, current: refresh_voices(current, mode), [voice_mode, voice], voice, queue=False)
            voice.change(load_library_voice, [voice, voice_mode], [single_audio, multiple_audio, voice_transcript, voice_language, voice_status], queue=False)
            voice_refresh.click(lambda mode, current: refresh_voices(current, mode), [voice_mode, voice], voice, queue=False)
            voice_refresh.click(load_library_voice, [voice, voice_mode], [single_audio, multiple_audio, voice_transcript, voice_language, voice_status], queue=False)
            voice_delete.click(delete_voice, [voice, voice_mode], [voice, single_audio, multiple_audio, voice_transcript, voice_language, voice_status], queue=False)
            save_voice.click(save_voice_ui, [single_audio, multiple_audio, voice_name, voice_transcript, voice_language, voice_mode], [voice_status, voice], queue=False)
            transcribe.click(transcribe_reference, [single_audio, whisper_ref], [voice_transcript, voice_status], queue=False)
            transcribe_stop.click(B.stop_aux_job, outputs=voice_status, queue=False)

        with gr.Tab("🔊 Inference"):
            gr.Markdown("*Use the Qwen base family or any compatible completed fine-tune through the same voice, conditioning and chunking workflow.*", elem_classes="tab-subtitle")
            with gr.Accordion("📖 Quick Guide", open=False):
                    gr.Markdown("""**Model and workflow** — Select a Qwen checkpoint to load its supported workflow. **CustomVoice** provides the bundled speaker list and optional style instructions. **VoiceDesign** uses a written voice description. **Base / Voice Clone** uses one reference or a multiple-reference package. A trained hybrid checkpoint appears as **Unofficial Hybrid Trained** while using the normal Base / Voice Clone controls.

**Unofficial Hybrid Trained variant** — This complete trained checkpoint is a hybrid: with external reference audio, the existing boolean keeps the current ICL/Speaker-only behavior. New exports also bundle a canonical ICL prompt and the trained speaker x-vector, so when no audio/transcript pair is supplied the same boolean selects internal ICL or Speaker-only generation. You may still provide reference audio when you want to clone a different voice. It does not accept a separate LoRA adapter and may behave differently from an official Qwen checkpoint.

**Reference conditioning** — Select a Voice Library entry or upload/record reference audio when using an external voice. External **ICL** requires the exact reference transcript. **Speaker-only mode** uses speaker information without transcript text. A new trained artifact can use its bundled ICL prompt or bundled speaker prompt when no external audio/transcript pair is supplied.

**Instruction Library** — Select a saved style or delivery prompt to load it into the active instruction field. The text remains editable before generation.

**Generation settings** — Temperature, Top-P, Top-K and Repetition Penalty control sampling. **Max New Tokens** is the native Qwen codec-token limit per chunk; the progress description and Live Console report generated tokens / limit, percentage and `it/s` after each native generation call. **Split / Chunking Rule** controls long text, while **Chunk Gap** adds silence between chunks. Fixed Seed can be enabled for repeatable test conditions.

**Runtime** — Standard Qwen supports the full workflow family, multiple references and compatible PEFT/LoRA adapters. Faster Qwen CUDA Graphs loads complete compatible 12 Hz checkpoints directly, including official CustomVoice and complete hybrid trained artifacts. A Base + LoRA adapter can run live with Standard Qwen; Faster Qwen requires **Merge LoRA for Fast Inference** because it does not apply a separate adapter at graph-capture time. Packaged speaker-only and internal ICL prompts are copied during merge, so a merged PEFT artifact preserves both no-reference paths.""")
            with gr.Row(elem_classes="toolbar"):
                model = gr.Dropdown(models(), value=initial_model_choice, label="Model", info="Official entries use the Qwen model catalog. Local entries are compatible checkpoints found on disk; downloaded catalog checkpoints are not listed twice.", scale=4)
                model_refresh = gr.Button("↻", elem_classes="compact")
                trained_checkpoint = gr.Dropdown(initial_checkpoint_choices, value=initial_checkpoint_choices[0][1] if initial_checkpoint_choices else None, label="Trained Checkpoint", info="Select the published model or an intermediate training checkpoint for inference.", visible=len(initial_checkpoint_choices) > 1, scale=4)
                active_model = gr.State(initial_checkpoint_choices[0][1] if initial_checkpoint_choices else initial_model_choice)
                mode = gr.Dropdown(WORKFLOW_CHOICES, value=initial_workflow, label="Workflow", info="Determined by the selected Qwen checkpoint.", interactive=False, scale=3)
                language = gr.Dropdown(B.LANGUAGE_CHOICES, value="Auto", label="Language", info="Language sent to Qwen; Auto enables model language detection.", scale=2)
            model_capability_status = gr.Markdown(model_capability_details(initial_model_choice), elem_classes="compact-status")
            gr.Markdown("### ⚙️ Runtime", elem_classes="section-heading")
            with gr.Row(elem_classes="runtime-row"):
                engine = gr.Dropdown(B.INFERENCE_ENGINE_CHOICES, value=B.DEFAULT_INFERENCE_ENGINE, label="Inference Engine", info="Qwen talker, Base speaker encoder and neural speech tokenizer use cuda:0 when available. CPU remains expected for file I/O, waveform decoding/resampling, frontend feature preparation and WAV serialization.", scale=3)
                adapter = gr.Dropdown(loras(initial_model_choice), value=NONE, label="LoRA Adapter", scale=4, visible=initial_lora_visible)
                adapter_refresh = gr.Button("↻", elem_classes="compact", visible=initial_lora_visible)
                lora_scale = gr.Slider(0, 1.5, 1.0, .05, label="LoRA Strength", scale=2, visible=initial_lora_visible)
                merge_lora = gr.Button("Merge LoRA for Fast Inference", variant="secondary", scale=3, visible=initial_lora_visible)
            merge_status = gr.Markdown("Base + LoRA runs live only with Standard Qwen. Merge the adapter into a complete checkpoint before using Faster Qwen CUDA Graphs.", elem_classes="compact-status", visible=initial_lora_visible)
            with gr.Accordion("🎛️ Generation Settings", open=True):
                with gr.Row():
                    temperature = gr.Slider(.1, 1.5, .9, .05, label="Temperature")
                    top_p = gr.Slider(.1, 1, 1.0, .05, label="Top-P")
                    top_k = gr.Slider(1, 100, 50, 1, label="Top-K")
                    repetition = gr.Slider(1, 2, 1.05, .01, label="Repetition Penalty")
                with gr.Row():
                    max_tokens = gr.Slider(128, 4096, 2048, 64, label="Max New Tokens", info="Maximum native Qwen codec tokens per text chunk. Progress reports generated tokens / this limit, percentage and it/s after each native generation call.")
                    chunk_mode = gr.Dropdown(B.CHUNK_CHOICES, value="None", label="Split / Chunking Rule")
                    chunk_gap = gr.Slider(0, 2, .25, .05, label="Chunk Gap (s)")
                    attention = gr.Dropdown(B.ATTENTION_CHOICES, value=B.ATTENTION_CHOICES[0], label="Attention Fallback")
                with gr.Row(elem_classes="seed-layout"):
                    seed = gr.Number(0, precision=0, minimum=0, maximum=(1 << 31) - 1, label="Seed")
                    fixed_seed = gr.Checkbox(False, label="Enable Fixed Seed")
                    last_seed = gr.State(0)
                    reuse_seed = gr.Button("Reuse Last Seed", variant="secondary", elem_classes="seed-action")
                    random_seed = gr.Button("Random Seed", variant="secondary", elem_classes="seed-action")
            with gr.Tabs():
                with gr.Tab("Single Inference"):
                    with gr.Row():
                        with gr.Column(scale=1, visible=initial_workflow == "Base / Voice Clone") as reference_section:
                            reference_mode = gr.Dropdown(["Single audio reference", "Multiple audio references"], value="Single audio reference", label="Audio Reference Mode")
                            with gr.Row(elem_classes="voice-library-row"):
                                inf_voice = gr.Dropdown(voice_library_choices("Single reference mode"), value=NONE, label="Voice Library", info="Reusable Qwen references filtered by Audio Reference Mode. 📦 marks a multiple-reference package; 🎙️ marks a single reference.", scale=8)
                                inf_voice_refresh = gr.Button("↻", elem_classes="compact")
                            ref_audio = gr.Audio(type="filepath", sources=["upload", "microphone"], label="Reference Audio / Preview", elem_classes="audio-safe-space")
                            multiple_refs = gr.File(file_count="multiple", file_types=["audio"], type="filepath", label="Multiple Reference Audio (native multi-file)", visible=False)
                            ref_text = gr.Textbox(label="Reference Transcript", lines=4, info="Exact text required for external ICL. For multiple references, use matching text or Speaker-only mode. A new trained hybrid checkpoint can use its bundled ICL prompt when no external audio/transcript pair is supplied.")
                            x_vector_only = gr.Checkbox(False, label="Speaker-only mode (no transcript)", info="With external reference audio, this preserves the current speaker-only path. If no audio/transcript pair is supplied, the packaged trained artifact uses this boolean to choose Speaker-only instead of its internal ICL prompt.")
                            with gr.Accordion("🛰️ Faster-Whisper Transcription", open=False):
                                infer_whisper = gr.Dropdown(list(B.WHISPER_MODELS), value=B.DEFAULT_WHISPER_MODEL, label="Whisper Model", info="Optional Faster-Whisper model used only to draft the reference transcript.")
                                with gr.Row(elem_classes="workflow-actions"):
                                    infer_transcribe = gr.Button("Transcribe Reference", variant="secondary", scale=5)
                                    infer_transcribe_stop = gr.Button("🛑 Stop", variant="stop", scale=1)
                            ref_status = gr.Markdown("No saved voice selected.", elem_classes="compact-status")
                        with gr.Column(scale=2):
                            with gr.Row():
                                speaker = gr.Dropdown(B.get_supported_speakers(initial_model_choice) or B.SPEAKERS, value=(B.get_supported_speakers(initial_model_choice) or B.SPEAKERS)[0], label=initial_speaker_label, info=initial_speaker_info, scale=2, visible=initial_workflow in {"CustomVoice", "Trained Custom Voice"})
                            with gr.Accordion("📝 Instruction · 📚 Saved Instruction Library", open=False, visible=initial_workflow in INSTRUCTION_WORKFLOW_MODES) as instruction_library:
                                instruct = gr.Textbox(label=initial_instruction_label, placeholder="e.g. warm, intimate, expressive and slightly slower", info=initial_instruction_info, visible=initial_workflow in INSTRUCTION_WORKFLOW_MODES)
                                with gr.Row(elem_classes="voice-library-row"):
                                    instruction_choice = gr.Dropdown(instruction_choices(), value=NONE, label="Load Saved Instruction", info="Loads text into the instruction field above.", scale=7)
                                    instruction_refresh = gr.Button("↻", elem_classes="compact")
                                    instruction_delete = gr.Button("🗑️", variant="stop", elem_classes="compact")
                                with gr.Row():
                                    instruction_name = gr.Textbox(label="Instruction Name", placeholder="warm_narration", scale=3)
                                    instruction_save = gr.Button("Save Instruction", variant="secondary", scale=2)
                                instruction_status = gr.Markdown(elem_classes="compact-status")
                            text = gr.Textbox(label="Text", lines=8, placeholder="Write the text you want Qwen to speak.")
                            with gr.Row(elem_classes="workflow-actions"):
                                generate = gr.Button("Generate Audio 🚀", variant="primary", size="lg")
                                inference_stop = gr.Button("🛑 Stop", variant="stop", size="lg", interactive=False)
                            output_audio = gr.Audio(type="filepath", label="Generated Audio", elem_classes="output-clean")
                            inference_status = gr.Markdown()
                    inf_voice.change(lambda name, rm: load_library_voice(name, "Multiple reference mode" if rm == "Multiple audio references" else "Single reference mode"), [inf_voice, reference_mode], [ref_audio, multiple_refs, ref_text, language, ref_status], queue=False)
                    inf_voice_refresh.click(lambda rm, current: refresh_voices(current, "Multiple reference mode" if rm == "Multiple audio references" else "Single reference mode"), [reference_mode, inf_voice], inf_voice, queue=False)
                    reference_mode.change(reference_visibility, reference_mode, [ref_audio, multiple_refs], queue=False)
                    reference_mode.change(reference_mode_changed, [reference_mode, inf_voice], [inf_voice, ref_audio, multiple_refs, ref_text, language, ref_status], queue=False)
                    mode.change(mode_visibility, mode, [speaker, instruct, reference_section, ref_audio, ref_text, x_vector_only, instruction_library], queue=False)
                    mode.change(lora_visibility, [mode, model, adapter], [adapter, adapter_refresh, lora_scale, merge_lora, merge_status], queue=False)
                    infer_transcribe.click(transcribe_reference, [ref_audio, infer_whisper], [ref_text, ref_status], queue=False)
                    infer_transcribe_stop.click(B.stop_aux_job, outputs=ref_status, queue=False)
                    instruction_choice.change(load_instruction_ui, instruction_choice, instruct, queue=False)
                    instruction_refresh.click(refresh_instructions, instruction_choice, instruction_choice, queue=False)
                    instruction_delete.click(delete_instruction_ui, instruction_choice, [instruction_choice, instruction_status], queue=False)
                    instruction_save.click(save_instruction_ui, [instruction_name, instruct, instruction_choice], [instruction_choice, instruction_status], queue=False)
                    inference_event = generate.click(
                        lambda: (gr.update(value="Generating Audio...", interactive=False), gr.update(interactive=True)),
                        outputs=[generate, inference_stop],
                        queue=False,
                    ).then(
                        run_inference,
                        [active_model, mode, text, language, speaker, instruct, inf_voice, reference_mode, ref_audio, multiple_refs, ref_text, x_vector_only, adapter, lora_scale, temperature, top_p, top_k, repetition, max_tokens, chunk_mode, chunk_gap, attention, engine, seed, fixed_seed],
                        [output_audio, inference_status, seed, last_seed],
                        show_progress="full",
                    ).then(
                        lambda: (gr.update(value="Generate Audio 🚀", interactive=True), gr.update(interactive=False)),
                        outputs=[generate, inference_stop],
                        queue=False,
                    )
                    inference_stop.click(B.stop_inference, outputs=inference_status, queue=False)
                with gr.Tab("Dialogue Builder"):
                    gr.Markdown("*Each row is a Qwen speaker turn. CustomVoice uses a bundled speaker, Trained Custom Voice uses the trained speaker, VoiceDesign uses the shared description, and Base / Voice Clone uses the Voice Library reference selected in that row.*", elem_classes="tab-subtitle")
                    with gr.Accordion("📖 Quick Guide", open=False):
                        gr.Markdown("""Select a model above and write one line per turn. **CustomVoice** turns use a bundled speaker, **Trained Custom Voice** turns use the speaker available in the checkpoint, **VoiceDesign** turns use the shared voice description, and **Base / Voice Clone** turns use the Voice Library reference selected in that row.

**Insert** creates a blank turn after the selected row, **Clone** duplicates its text and voice selection, and **Remove** deletes the row. The shared language, generation settings, runtime and attention selection apply to every turn. **Chunk Gap** affects long text within one turn; **Dialogue Silence** affects the space between turns.

For Base / Voice Clone, choose **Single audio reference** or **Multiple audio references** to filter the Voice Library entries available in every row. Each row uses its own saved Voice Library entry, including its transcript and reference-file list. Dialogue Builder does not expose packaged no-reference prompts; use Single Inference when you need a trained artifact's internal speaker or ICL prompt.""")
                    with gr.Row(elem_classes="dialogue-toolbar"):
                        dialogue_mode = gr.Dropdown(WORKFLOW_CHOICES, value=initial_workflow, label="Dialogue Workflow", info="Determined by the selected Qwen checkpoint.", interactive=False, scale=3)
                        dialogue_reference_mode = gr.Dropdown(["Single audio reference", "Multiple audio references"], value="Single audio reference", label="Audio Reference Mode", info="Filters the Voice Library choices used by every Base / Voice Clone turn.", visible=initial_workflow == "Base / Voice Clone", scale=2)
                        dialogue_silence = gr.Slider(0, 2, .35, .05, label="Dialogue Silence (s)", info="Silence inserted between generated turns; independent from Chunk Gap.", scale=2)
                        dialogue_voice_refresh = gr.Button("↻ Refresh Voice Library", variant="secondary", scale=2)
                    with gr.Accordion("📝 Instruction (optional) · 📚 Saved Instruction Library", open=False, visible=initial_workflow in INSTRUCTION_WORKFLOW_MODES) as dialogue_instruction_library:
                        dialogue_instruct = gr.Textbox(label=initial_instruction_label, placeholder="e.g. natural, expressive and conversational", info=initial_instruction_info, visible=initial_workflow in INSTRUCTION_WORKFLOW_MODES)
                        with gr.Row(elem_classes="voice-library-row"):
                            dialogue_instruction_choice = gr.Dropdown(instruction_choices(), value=NONE, label="Load Saved Instruction", info="Loads text into the dialogue instruction field above.", scale=7)
                            dialogue_instruction_refresh = gr.Button("↻", elem_classes="compact")
                            dialogue_instruction_delete = gr.Button("🗑️", variant="stop", elem_classes="compact")
                        with gr.Row():
                            dialogue_instruction_name = gr.Textbox(label="Instruction Name", placeholder="friendly_dialogue", scale=3)
                            dialogue_instruction_save = gr.Button("Save Instruction", variant="secondary", scale=2)
                        dialogue_instruction_status = gr.Markdown(elem_classes="compact-status")
                    dialogue_count = gr.State(2)
                    dvoices, dtexts, drows, dinsert, dclone, dremove = [], [], [], [], [], []
                    for index in range(MAX_DIALOGUE):
                        with gr.Row(visible=index < 2, elem_classes="card") as row:
                            initial_dialogue_choices = dialogue_voice_choices(initial_workflow, initial_model_choice, "Single reference mode", NONE)
                            initial_dialogue_value = initial_dialogue_choices[0][1] if initial_dialogue_choices else NONE
                            dv = gr.Dropdown(initial_dialogue_choices, value=initial_dialogue_value, label=f"Turn {index + 1} · {dialogue_voice_label(initial_workflow)}", info="The label follows the selected Qwen workflow.", scale=2, visible=initial_workflow != "VoiceDesign")
                            dt = gr.Textbox(label="Text", lines=2, info="Only this speaker's line.", scale=6)
                            with gr.Row(elem_classes="dialogue-actions"):
                                ins = gr.Button("＋ Insert", size="sm"); cln = gr.Button("⧉ Clone", size="sm"); rem = gr.Button("− Remove", size="sm")
                        dvoices.append(dv); dtexts.append(dt); drows.append(row); dinsert.append(ins); dclone.append(cln); dremove.append(rem)
                    dialogue_outputs = [dialogue_count, *dvoices, *dtexts, *drows]
                    dialogue_inputs = [dialogue_count, *dvoices, *dtexts]
                    for index in range(MAX_DIALOGUE):
                        dinsert[index].click(lambda count, *values, _index=index: dialogue_mutate("insert", _index, count, *values), dialogue_inputs, dialogue_outputs, queue=False)
                        dclone[index].click(lambda count, *values, _index=index: dialogue_mutate("clone", _index, count, *values), dialogue_inputs, dialogue_outputs, queue=False)
                        dremove[index].click(lambda count, *values, _index=index: dialogue_mutate("remove", _index, count, *values), dialogue_inputs, dialogue_outputs, queue=False)
                    with gr.Row(elem_classes="workflow-actions"):
                        dialogue_generate = gr.Button("Generate Dialogue 🚀", variant="primary", size="lg")
                        dialogue_stop = gr.Button("🛑 Stop", variant="stop", size="lg", interactive=False)
                    dialogue_output = gr.Audio(type="filepath", label="Dialogue Output", elem_classes="output-clean"); dialogue_status = gr.Markdown()
                    dialogue_event = dialogue_generate.click(
                        lambda: (gr.update(value="Generating Dialogue...", interactive=False), gr.update(interactive=True)),
                        outputs=[dialogue_generate, dialogue_stop],
                        queue=False,
                    ).then(
                        run_dialogue,
                        [active_model, dialogue_mode, language, speaker, dialogue_instruct, adapter, lora_scale, temperature, top_p, top_k, repetition, max_tokens, chunk_mode, chunk_gap, dialogue_silence, attention, engine, seed, fixed_seed, dialogue_count, *dvoices, *dtexts],
                        [dialogue_output, dialogue_status, seed, last_seed],
                        show_progress="full",
                    ).then(
                        lambda: (gr.update(value="Generate Dialogue 🚀", interactive=True), gr.update(interactive=False)),
                        outputs=[dialogue_generate, dialogue_stop],
                        queue=False,
                    )
                    dialogue_stop.click(B.stop_inference, outputs=dialogue_status, queue=False)
                    dialogue_mode.change(refresh_dialogue_voices, [dialogue_mode, model, dialogue_reference_mode, adapter], dvoices, queue=False)
                    dialogue_mode.change(dialogue_mode_visibility, dialogue_mode, [dialogue_reference_mode, dialogue_instruction_library, dialogue_instruct], queue=False)
                    dialogue_reference_mode.change(refresh_dialogue_voices, [dialogue_mode, model, dialogue_reference_mode, adapter], dvoices, queue=False)
                    dialogue_instruction_choice.change(load_instruction_ui, dialogue_instruction_choice, dialogue_instruct, queue=False)
                    dialogue_instruction_refresh.click(refresh_instructions, dialogue_instruction_choice, dialogue_instruction_choice, queue=False)
                    dialogue_instruction_delete.click(delete_instruction_ui, dialogue_instruction_choice, [dialogue_instruction_choice, dialogue_instruction_status], queue=False)
                    dialogue_instruction_save.click(save_instruction_ui, [dialogue_instruction_name, dialogue_instruct, dialogue_instruction_choice], [dialogue_instruction_choice, dialogue_instruction_status], queue=False)
                    dialogue_voice_refresh.click(refresh_dialogue_voices, [dialogue_mode, model, dialogue_reference_mode, adapter], dvoices, queue=False)
                    adapter.change(refresh_dialogue_voices, [dialogue_mode, model, dialogue_reference_mode, adapter], dvoices, queue=False)
            inference_timer = gr.Timer(.5)
            inference_timer.tick(inference_poll_ui, outputs=[generate, inference_stop, dialogue_generate, dialogue_stop], queue=False, show_progress="hidden")
            model_refresh.click(refresh_models, model, model, queue=False).then(trained_checkpoint_update, model, [trained_checkpoint, active_model], queue=False)
            adapter_refresh.click(refresh_loras, [model, adapter], adapter, queue=False)
            merge_lora.click(merge_lora_ui, [model, adapter, lora_scale, attention], [model, adapter, merge_status]).then(trained_checkpoint_update, model, [trained_checkpoint, active_model], queue=False)
            trained_checkpoint.change(trained_checkpoint_selected, [trained_checkpoint, model], active_model, queue=False).then(refresh_dialogue_voices, [dialogue_mode, active_model, dialogue_reference_mode, adapter], dvoices, queue=False)
            model.change(model_changed, model, [mode, speaker, model_capability_status], queue=False).then(mode_visibility, mode, [speaker, instruct, reference_section, ref_audio, ref_text, x_vector_only, instruction_library], queue=False).then(lora_visibility, [mode, model, adapter], [adapter, adapter_refresh, lora_scale, merge_lora, merge_status], queue=False).then(dialogue_workflow_update, model, dialogue_mode, queue=False).then(dialogue_mode_visibility, dialogue_mode, [dialogue_reference_mode, dialogue_instruction_library, dialogue_instruct], queue=False).then(refresh_dialogue_voices, [dialogue_mode, model, dialogue_reference_mode, adapter], dvoices, queue=False).then(trained_checkpoint_update, model, [trained_checkpoint, active_model], queue=False)
            reuse_seed.click(lambda value: value or B.random_seed_value(), last_seed, seed, queue=False)
            random_seed.click(new_seed_pair, outputs=[seed, last_seed], queue=False)

        with gr.Tab("📂 Dataset Preparation"):
            gr.Markdown("*Turn a folder of speech plus TXT/LAB sidecars into Qwen 12 Hz JSONL training data. Faster-Whisper transcription and audio-code preparation run as part of Prepare Dataset.*", elem_classes="tab-subtitle")
            with gr.Accordion("📖 Quick Guide", open=False):
                gr.Markdown("""1. Create or select a **Dataset Project**. The project stores the source folder, language, transcription model and prepared-file paths.
2. Select the source folder. For each audio file, a same-stem `.txt`, `.lab` or `.transcript` sidecar is preferred. Faster-Whisper is used when a usable sidecar is not available.
3. Choose the language, Faster-Whisper model and ASR batch size. Lower the batch size if transcription approaches the available VRAM limit.
4. Press **Prepare Dataset**. The process normalizes audio, creates training/evaluation JSONL files, generates the Qwen 12 Hz audio codes and writes a dataset manifest.
5. After completion, the prepared dataset is refreshed and selected in Fine-tuning, and the matching Training Project is selected when available. Existing valid sidecar transcripts are reused.""")
            with gr.Row(elem_classes="project-strip"):
                dataset_project = gr.Dropdown([NONE, *list_projects("dataset")], value=NONE, label="Dataset Project", allow_custom_value=True, scale=3)
                dataset_new = gr.Textbox(label="New Project Name", scale=2); dataset_create = gr.Button("Create Project"); dataset_save = gr.Button("💾 Save Project"); dataset_clone = gr.Button("Clone"); dataset_refresh = gr.Button("↻", elem_classes="compact"); dataset_delete = gr.Button("🗑️", variant="stop", elem_classes="compact")
            dataset_status = gr.Markdown(elem_classes="status-card")
            with gr.Row():
                source_folder = gr.Textbox(label="Source Audio Folder", scale=6); source_browse = gr.Button("📁 Browse", elem_classes="compact")
                dataset_language = gr.Dropdown(B.LANGUAGE_CHOICES, value="Auto", label="Language", scale=2)
            with gr.Row():
                dataset_whisper = gr.Dropdown(list(B.WHISPER_MODELS), value=B.DEFAULT_WHISPER_MODEL, label="Whisper Model", info="Faster-Whisper model used only when a sidecar transcript is missing.", scale=3)
                dataset_batch = gr.Number(4, precision=0, minimum=1, maximum=32, label="ASR Batch Size", info="Lower this if transcription runs out of VRAM.", scale=2)
            with gr.Row(elem_classes="workflow-actions"):
                dataset_build = gr.Button("🧱 Prepare Dataset", variant="primary"); dataset_cancel = gr.Button("🛑 Cancel", variant="stop")
            train_raw = gr.Textbox(label="Train JSONL", interactive=False); eval_raw = gr.Textbox(label="Eval JSONL", interactive=False)
            dataset_create.click(lambda name: create_project_ui(name, "dataset"), dataset_new, [dataset_project, dataset_status], queue=False)
            dataset_save.click(save_dataset_project_ui, [dataset_project, source_folder, dataset_language, dataset_whisper], [dataset_project, dataset_status], queue=False)
            dataset_clone.click(lambda name: clone_project_ui(name, "dataset"), dataset_project, [dataset_project, dataset_status], queue=False)
            dataset_refresh.click(lambda: project_update("dataset"), outputs=dataset_project, queue=False)
            dataset_delete.click(lambda name: delete_project_ui(name, "dataset"), dataset_project, [dataset_project, dataset_status], queue=False)
            dataset_project.change(load_dataset_project, dataset_project, [source_folder, dataset_language, dataset_whisper], queue=False)
            source_browse.click(browse_folder, source_folder, source_folder, queue=False)
            dataset_build_event = dataset_build.click(prepare_dataset_ui, [source_folder, dataset_project, dataset_language, dataset_whisper, dataset_batch], [dataset_status, train_raw, eval_raw, dataset_project])
            dataset_cancel.click(B.stop_aux_job, outputs=dataset_status, queue=False)

        with gr.Tab("🚀 Fine-tuning"):
            gr.Markdown("*Train a complete Qwen 12 Hz checkpoint or create a PEFT/LoRA adapter for compatible Standard Qwen inference.*", elem_classes="tab-subtitle")
            gr.Markdown("**Training note:** Qwen TTS fine-tuning is experimental. Small or low-quality datasets may provide no noticeable benefit. Use it when you need to address a limitation in the base model's language coverage, with a generously sized dataset—from 30 minutes to several hours. The Qwen base model is already strongly fine-tuned and performs well in most cases. AutoTune is a practical starting point, not a guarantee; increase the suggested epochs or evaluation cadence when your dataset requires it. A recommended workflow is to choose a generous round number such as **50 epochs**, save and evaluate every **10 epochs**, and keep the best-performing checkpoint. Review TensorBoard regularly to follow the training evolution and compare checkpoints.", elem_classes="compact-status")
            with gr.Accordion("📖 Quick Guide", open=False):
                gr.Markdown("""1. Select a prepared dataset. **Prepare Dataset** creates the JSONL files and generates the Qwen 12 Hz audio codes required by training.
2. Select a compatible 12 Hz Base model and choose the output. Full SFT always produces one complete **hybrid Base-compatible checkpoint**: with an external audio/transcript pair, the existing boolean selects ICL or Speaker-only conditioning; new exports also package one canonical ICL prompt and one speaker prompt for use when no external pair is supplied. This keeps one trained model for both behaviors and avoids duplicate exports. **PEFT LoRA adapter** produces a Base-compatible adapter for Standard Qwen; new adapters package the same optional no-reference prompts, while its Evaluation Zone always evaluates with the fixed external reference audio and exact transcript. It can later be merged once for CUDA-graph inference. Use **Resume Checkpoint** to continue the selected model or adapter from its saved epoch position; set **Epochs** above that checkpoint's epoch. Leave `Fresh / None` to reset this Training Project's generated outputs and start from the Base model. Model-only checkpoints do not include optimizer moments, so the optimizer state is recreated while the model weights, epoch counter and LR schedule position continue.
3. Enter the **Speaker Name** used by the dataset, select a **Hardware Preset** and press **AutoTune**. AutoTune reads train/eval counts, audio/code lengths and transcript lengths together with model size, output mode and VRAM profile before proposing the complete supported HP profile.
4. Review or edit epochs, batch size, gradient accumulation, learning-rate schedule, warmup/floor, save/evaluation cadence in epochs, weight decay, gradient clipping, seed, attention backend and LoRA rank/alpha/dropout. The recommended schedule warms up briefly and then decays the LR smoothly instead of forcing a high constant LR.
5. Optionally enable the **Evaluation Zone**. At the selected epoch interval, the worker saves the checkpoint first, then generates one comparable ICL WAV from the fixed external evaluation reference, its exact transcript and prepared codec codes for both Full SFT and LoRA. This deliberate external reference keeps checkpoint comparisons meaningful; it does not use the packaged no-reference prompt. Training resumes after the sample finishes; the audio, conditioning mode, generation time and training loss are written to the project-local TensorBoard log. The optional Faster Qwen CUDA-graph switch is off by default and falls back to Standard Qwen when the model or graph capture is unsupported.
6. Start training and monitor the progress panel or TensorBoard. The current Qwen training format is single-speaker; compare the resulting checkpoint against Base / Voice Clone using the same reference and evaluation text.""")
            with gr.Row(elem_classes="project-strip"):
                training_project = gr.Dropdown([NONE, *list_projects("training")], value=NONE, label="Training Project", allow_custom_value=True, scale=3)
                training_new = gr.Textbox(label="New Project Name", scale=2); training_create = gr.Button("Create Project"); training_save = gr.Button("💾 Save Project"); training_clone = gr.Button("Clone"); training_refresh = gr.Button("↻", elem_classes="compact"); training_delete = gr.Button("🗑️", variant="stop", elem_classes="compact")
            training_status = gr.Markdown(elem_classes="status-card")
            with gr.Row():
                training_resume = gr.Dropdown(B.training_resume_choices(NONE), value=B.FRESH_RESUME, label="Resume Checkpoint", scale=6, info="Fresh / None starts from the selected Base model and resets this project's generated outputs. A listed checkpoint resumes its model/adapter weights and epoch position; set Epochs above the checkpoint epoch. Optimizer moments are not stored in these model-only checkpoints.")
                training_resume_refresh = gr.Button("↻", elem_classes="compact")
            with gr.Row():
                training_dataset = gr.Dropdown([NONE, *B.list_datasets()], value=B.list_datasets()[0] if B.list_datasets() else NONE, label="Prepared Dataset", scale=3)
                training_base = gr.Dropdown([(label, label) for label in B.MODEL_CATALOG if "Base / Voice Clone" in label and label.startswith("12Hz")], value=next(label for label in B.MODEL_CATALOG if "1.7B" in label and "Base / Voice Clone" in label), label="12 Hz Base Model", info="Only official 12 Hz Base checkpoints are supported by Qwen's training scripts; the label includes approximate VRAM.", scale=3)
                training_mode = gr.Dropdown(["Qwen-compatible Full SFT", "PEFT LoRA adapter (experimental)"], value="Qwen-compatible Full SFT", label="Training Output", info="Full SFT produces one hybrid Base-compatible checkpoint for ICL and Speaker-only inference. PEFT produces a separate LoRA adapter.", scale=3)
                training_speaker = gr.Textbox(value="speaker_1", label="Speaker Name", scale=2)
                training_model_export_mode = gr.Dropdown(["ICL-variant"], value="ICL-variant", label="Model Export (fixed hybrid ICL)", info="Full SFT produces one Base-compatible hybrid checkpoint. With an external reference pair, the inference boolean selects ICL or Speaker-only conditioning; without that pair, new exports use their bundled ICL or speaker prompt.", interactive=False, visible=False, scale=3)
            with gr.Row():
                training_hardware = gr.Dropdown(B.TRAINING_HARDWARE_PRESETS, value="24 GB VRAM (1.7B minimum)", label="Hardware Preset", info="Conservative VRAM budget used by AutoTune; it does not reserve VRAM or bypass an out-of-memory error.", scale=3)
                training_autotune = gr.Button("⚙ AutoTune", variant="secondary", scale=1)
                training_autotune_status = gr.Markdown("Select a dataset and hardware preset, then press AutoTune.", elem_classes="compact-status")
            with gr.Row():
                training_epochs = gr.Slider(1, 100, 3, 1, label="Epochs", info="Full passes over the prepared training JSONL.")
                training_batch = gr.Slider(1, 32, 1, 1, label="Batch Size", info="Samples per optimizer micro-batch; limited by model VRAM.")
                training_grad = gr.Slider(1, 64, 4, 1, label="Gradient Accumulation", info="Micro-batches accumulated before one optimizer update.")
                training_lr = gr.Number(2e-6, label="Learning Rate", info="Conservative AdamW learning rate for the Qwen codec autoregressive stack; lower it further if the loss becomes non-finite.")
            with gr.Row():
                training_lr_schedule = gr.Dropdown(B.TRAINING_LR_SCHEDULE_CHOICES, value="warmup_cosine", label="LR Schedule", info="Warmup + cosine smoothly reduces the learning rate while retaining a non-zero floor. Constant is retained for diagnostics and legacy comparisons.")
                training_warmup_ratio = gr.Number(.05, minimum=0, maximum=.25, label="Warmup Ratio", info="Fraction of optimizer steps used to ramp from 10% to the selected peak LR.")
                training_min_lr_ratio = gr.Number(.20, minimum=.01, maximum=1, label="Minimum LR Ratio", info="Final LR as a fraction of the selected peak LR; 0.20 keeps late training active without an aggressive LR.")
            with gr.Row():
                training_save_every = gr.Slider(0, 100, 0, step=1, label="Save Every (Epochs)", info="Checkpoint cadence in complete epochs; 0 disables intermediate checkpoints. The final epoch is always saved.")
                training_weight_decay = gr.Number(.01, label="Weight Decay", info="AdamW decoupled weight decay.")
                training_max_grad_norm = gr.Number(1.0, label="Max Gradient Norm", info="Gradient clipping threshold.")
                training_seed = gr.Number(1234, precision=0, label="Training Seed", info="Seed passed to Python, PyTorch and CUDA before training.")
            training_eval_steps = gr.Slider(1, 100, 1, step=1, label="Evaluate Every (Epochs)", info="When Evaluation Zone is enabled, generate the fixed monitoring sample every N complete epochs and log it to TensorBoard. The final epoch is always evaluated.")
            with gr.Accordion("🧩 LoRA, attention and advanced settings", open=False, visible=False) as training_advanced_zone:
                with gr.Row():
                    training_rank = gr.Slider(1, 128, 16, 1, label="LoRA Rank", info="Adapter rank; used only for PEFT LoRA output.")
                    training_alpha = gr.Slider(1, 256, 32, 1, label="LoRA Alpha", info="Adapter scaling factor; used only for PEFT LoRA output.")
                    training_dropout = gr.Slider(0, .5, .05, .01, label="LoRA Dropout", info="Adapter dropout; used only for PEFT LoRA output.")
                    training_attention = gr.Dropdown(B.TRAINING_ATTENTION_CHOICES, value=B.TRAINING_ATTENTION_CHOICES[0], label="Attention Fallback", info="Auto uses SDPA → eager for training stability. FlashAttention 2 remains available as an explicit opt-in.")
            training_eval_enabled = gr.Checkbox(False, label="Enable Evaluation Zone", info="Optional prompted monitoring audio. A fixed evaluation reference and sentence are reused across checkpoints.")
            with gr.Accordion("🎧 Prompted Holdout Evaluation", open=False, visible=False) as training_eval_zone:
                gr.Markdown("At **Evaluate Every (Epochs)**, the worker saves the checkpoint first and generates the same fixed sentence with Base ICL: reference audio, its exact transcript and codec codes are used for both Full SFT and LoRA. Training pauses during the monitoring clip and resumes when it is complete. The WAV, conditioning mode, generation time and training loss are logged to TensorBoard. If the dataset has no holdout, the first deterministic training sample is used and the run is labeled accordingly.", elem_classes="compact-status")
                training_eval_text = gr.Textbox(label="Fixed Eval Text", lines=4, value="This is a fixed evaluation sample generated during training.", info="Keep this sentence unchanged during a run so checkpoint comparisons remain meaningful.")
                training_eval_fast = gr.Checkbox(False, label="Use Faster Qwen CUDA Graphs for Evaluation", info="Optional and off by default. Reuses the full SFT model already in memory; PEFT/LoRA, unsupported layouts or capture failures fall back to Standard Qwen automatically.")
                training_eval_icl = gr.Checkbox(True, label="Use ICL for Evaluation (fixed)", info="Always enabled internally for the Evaluation Zone. The control is retained hidden for compatibility with older project files.", visible=False)
                training_eval_status = gr.Markdown("The reference is selected automatically from the prepared evaluation JSONL.", elem_classes="compact-status")
            training_eval_enabled.change(lambda value: gr.update(visible=bool(value)), training_eval_enabled, training_eval_zone, queue=False)
            # Model Export is a fixed hidden compatibility control.  Do not
            # send its stale browser value as an event input; Gradio validates
            # hidden dropdown values before the handler can replace them.
            training_mode.change(model_export_visibility, training_mode, training_model_export_mode, queue=False)
            training_mode.change(training_advanced_visibility, training_mode, training_advanced_zone, queue=False)
            training_mode.change(training_eval_icl_visibility, [training_mode, training_eval_icl], training_eval_icl, queue=False)
            with gr.Row(elem_classes="workflow-actions"):
                training_start = gr.Button("🚀 Start Training", variant="primary"); training_stop = gr.Button("🛑 Stop Training", variant="stop", interactive=False); tensorboard = gr.Button("📊 Open TensorBoard", variant="secondary")
            training_progress = gr.HTML(training_progress_html())
            training_console_status = gr.Markdown()
            training_timer = gr.Timer(1)
            training_timer.tick(training_poll_ui, outputs=[training_progress, training_start, training_stop], queue=False, show_progress="hidden")
            training_create.click(lambda name: create_project_ui(name, "training"), training_new, [training_project, training_status], queue=False)
            training_save.click(save_training_project_ui, [training_project, training_dataset, training_base, training_mode, training_speaker, training_hardware, training_epochs, training_batch, training_grad, training_lr, training_lr_schedule, training_warmup_ratio, training_min_lr_ratio, training_save_every, training_eval_steps, training_rank, training_alpha, training_dropout, training_weight_decay, training_max_grad_norm, training_seed, training_attention, training_resume, training_eval_enabled, training_eval_text, training_eval_icl, training_eval_fast], [training_project, training_status], queue=False)
            training_clone.click(lambda name: clone_project_ui(name, "training"), training_project, [training_project, training_status], queue=False)
            training_refresh.click(lambda: project_update("training"), outputs=training_project, queue=False)
            training_delete.click(lambda name: delete_project_ui(name, "training"), training_project, [training_project, training_status], queue=False)
            training_resume_refresh.click(refresh_training_resume_ui, [training_project, training_mode, training_resume], training_resume, queue=False)
            training_mode.change(refresh_training_resume_ui, [training_project, training_mode, training_resume], training_resume, queue=False)
            training_project.change(load_training_project, training_project, [training_dataset, training_base, training_mode, training_speaker, training_hardware, training_epochs, training_batch, training_grad, training_lr, training_lr_schedule, training_warmup_ratio, training_min_lr_ratio, training_save_every, training_eval_steps, training_rank, training_alpha, training_dropout, training_weight_decay, training_max_grad_norm, training_seed, training_attention, training_resume, training_eval_enabled, training_eval_text, training_eval_icl, training_eval_fast, training_eval_zone, training_model_export_mode], queue=False).then(training_advanced_visibility, training_mode, training_advanced_zone, queue=False)
            training_autotune.click(autotune_training_ui, [training_dataset, training_base, training_mode, training_hardware], [training_epochs, training_batch, training_grad, training_lr, training_lr_schedule, training_warmup_ratio, training_min_lr_ratio, training_save_every, training_eval_steps, training_rank, training_alpha, training_dropout, training_weight_decay, training_max_grad_norm, training_seed, training_attention, training_autotune_status], queue=False)
            training_start.click(lambda: (gr.update(value="Preparing...", interactive=False), gr.update(interactive=False)), outputs=[training_start, training_stop], queue=False).then(start_training_ui, [training_project, training_dataset, training_base, training_mode, training_speaker, training_hardware, training_epochs, training_batch, training_grad, training_lr, training_lr_schedule, training_warmup_ratio, training_min_lr_ratio, training_save_every, training_eval_steps, training_rank, training_alpha, training_dropout, training_weight_decay, training_max_grad_norm, training_seed, training_attention, training_resume, training_eval_enabled, training_eval_text, training_eval_icl, training_eval_fast], training_console_status, queue=False)
            training_stop.click(B.stop_training, outputs=training_console_status, queue=False)
            tensorboard.click(open_tensorboard, training_project, training_console_status, queue=False)
            dataset_build_event.then(sync_dataset_completion, [dataset_project, training_project, training_dataset], [training_dataset, training_project], queue=False)

    with gr.Accordion("🖥️ Live Console & Training Log", open=True, elem_classes="console-accordion"):
        console = gr.HTML(value=html_view("Qwen TTS Easy GUI Live Console")); console_timer = gr.Timer(.5)
        console_timer.tick(lambda: html_view("Qwen TTS Easy GUI Console"), outputs=console, queue=False, show_progress="hidden")


if __name__ == "__main__":
    app.queue().launch(inbrowser=True, server_name="127.0.0.1", show_error=True)
