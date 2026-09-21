"""Prompted evaluation helpers shared by the Qwen 12 Hz training workers.

The official Qwen fine-tuning scripts do not expose a validation synthesizer.
This module keeps the optional monitoring path deliberately small and
deterministic: one JSONL reference, one fixed sentence and one TensorBoard
audio sample per requested epoch interval.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Optional


DEFAULT_EVAL_TEXT = "This is a fixed evaluation sample generated during training."
# Qwen 12 Hz produces one codec frame per 1/12 second.  Evaluation is a
# monitoring clip, not the final inference request.  Keep it bounded so a
# checkpoint that does not emit EOS cannot consume the whole run.
EVAL_MAX_NEW_TOKENS = 128

def load_eval_record(*paths: str | Path | None) -> tuple[Optional[dict[str, Any]], str]:
    """Return the first usable record and the source file used.

    The evaluation JSONL is preferred, while the training JSONL is a safe
    deterministic fallback for tiny datasets without an independent holdout.
    """
    candidates: list[Path] = []
    for raw_path in paths:
        if raw_path:
            path = Path(str(raw_path))
            if path not in candidates:
                candidates.append(path)
    for path in candidates:
        if not path.is_file():
            continue
        records: list[dict[str, Any]] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                audio = str(item.get("ref_audio") or item.get("audio") or "").strip()
                text = str(item.get("text") or "").strip()
                if audio and text and Path(audio).is_file():
                    records.append(item)
        except Exception:
            continue
        if records:
            record = sorted(records, key=lambda value: str(value.get("audio") or value.get("ref_audio") or "").casefold())[0]
            return record, str(path)
    return None, ""


def create_tensorboard_writer(output_dir: str | Path, config: dict[str, Any]):
    """Create the project-local SummaryWriter used by both training modes."""
    from torch.utils.tensorboard import SummaryWriter

    log_dir = Path(output_dir) / "tensorboard"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir), flush_secs=5)
    writer.add_text("training/config", json.dumps(config, ensure_ascii=False, indent=2), 0)
    writer.flush()
    return writer


def write_evaluation_contract(
    output_dir: str | Path,
    *,
    enabled: bool,
    text: str,
    every_epochs: int,
    source: str,
    record: Optional[dict[str, Any]],
    fast_requested: bool = False,
    conditioning_mode: str = "ICL",
    icl_enabled: bool = False,
) -> Path:
    """Persist the fixed evaluation inputs beside the run's checkpoints."""
    payload = {
        "schema": 1,
        "enabled": bool(enabled),
        "text": str(text or DEFAULT_EVAL_TEXT),
        "every_epochs": int(max(0, every_epochs)),
        "source_jsonl": str(source or ""),
        "reference_audio": str((record or {}).get("ref_audio") or (record or {}).get("audio") or ""),
        "reference_text": str((record or {}).get("text") or ""),
        "reference_language": str((record or {}).get("language") or "Auto"),
        "conditioning_mode": str(conditioning_mode or "ICL"),
        "icl_enabled": bool(icl_enabled),
        "max_new_tokens": EVAL_MAX_NEW_TOKENS,
        "faster_qwen_requested": bool(fast_requested),
    }
    path = Path(output_dir) / "evaluation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def log_training_step(
    writer,
    *,
    step: int,
    epoch: int,
    loss: float,
    loss_ema: float | None = None,
    learning_rate: float | None = None,
    grad_norm: float | None = None,
) -> None:
    if writer is None:
        return
    writer.add_scalar("train/loss", float(loss), int(step))
    writer.add_scalar("train/epoch", int(epoch), int(step))
    if loss_ema is not None:
        writer.add_scalar("train/loss_ema", float(loss_ema), int(step))
    if learning_rate is not None:
        writer.add_scalar("train/learning_rate", float(learning_rate), int(step))
    if grad_norm is not None:
        writer.add_scalar("train/grad_norm", float(grad_norm), int(step))
    writer.flush()


def log_training_epoch(writer, *, epoch: int, loss_mean: float, step: int) -> None:
    """Write an epoch aggregate so noisy batch loss is not the only signal."""
    if writer is None:
        return
    writer.add_scalar("train/epoch_loss", float(loss_mean), int(step))
    writer.flush()


def place_speech_tokenizer_on_device(qwen3tts, device, log=print) -> bool:
    """Place Qwen's codec decoder beside the trained talker before eval.

    ``Accelerator.prepare`` receives only ``qwen3tts.model`` in the training
    workers.  Qwen keeps the speech tokenizer as a separate wrapper, so its
    decoder otherwise remains on CPU even when the talker is on CUDA.  The
    standalone inference loader uses ``device_map`` for both components,
    which is why it does not show the multi-minute decode seen in training.
    """
    import torch

    if device in (None, "", "unknown"):
        return False
    tokenizer = getattr(getattr(qwen3tts, "model", None), "speech_tokenizer", None)
    tokenizer_model = getattr(tokenizer, "model", None)
    if tokenizer_model is None or not hasattr(tokenizer_model, "parameters"):
        log("[EVAL] Speech tokenizer device could not be inspected; leaving it unchanged.")
        return False
    target_device = torch.device(device)
    try:
        first_parameter = next(tokenizer_model.parameters())
        source_device = first_parameter.device
    except StopIteration:
        source_device = target_device
    try:
        if source_device != target_device:
            tokenizer_model.to(target_device)
        tokenizer.device = target_device
        tokenizer_model.eval()
        log(
            f"[EVAL] Speech tokenizer placed on {target_device} "
            f"(was {source_device}); codec decode will use the training device."
        )
        return True
    except Exception as exc:
        log(
            f"[EVAL] Could not move speech tokenizer from {source_device} to {target_device}; "
            f"evaluation may be CPU-bound: {exc}"
        )
        return False


def _build_faster_eval_model(*, qwen3tts, model, model_device):
    """Build CUDA-graph wrappers around the already loaded full SFT model.

    Faster Qwen's public ``from_pretrained`` path loads another complete model,
    which is undesirable inside a training worker.  Its wrapper is also able
    to receive the existing Qwen model and graph objects, so evaluation can
    reuse the current weights without a second model copy.  PEFT models are
    rejected here because their adapter dispatch is not validated by the
    CUDA-graph implementation; the caller falls back to Standard Qwen.
    """
    import torch

    device = torch.device(model_device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Faster Qwen evaluation requires a CUDA device.")
    if model is None or not hasattr(model, "talker") or not hasattr(model, "config"):
        raise RuntimeError("The current training model is not a complete Qwen module.")

    model_module = type(model).__module__.casefold()
    model_name = type(model).__name__.casefold()
    if "peft" in model_module or "peft" in model_name or hasattr(model, "peft_config"):
        raise RuntimeError("PEFT/LoRA models remain on Standard Qwen for evaluation.")

    try:
        parameter_dtype = next(model.parameters()).dtype
    except StopIteration as exc:
        raise RuntimeError("The current Qwen model has no parameters.") from exc
    if parameter_dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise RuntimeError(f"Unsupported Qwen dtype for CUDA graphs: {parameter_dtype}.")

    from faster_qwen3_tts import FasterQwen3TTS
    from faster_qwen3_tts.predictor_graph import PredictorGraph
    from faster_qwen3_tts.talker_graph import TalkerGraph

    talker = model.talker
    talker_config = model.config.talker_config
    predictor = talker.code_predictor
    pred_config = predictor.model.config
    talker_hidden = talker_config.hidden_size
    device_name = str(device)

    predictor_graph = PredictorGraph(
        predictor,
        pred_config,
        talker_hidden,
        device=device_name,
        dtype=parameter_dtype,
        do_sample=False,
    )
    talker_graph = TalkerGraph(
        talker.model,
        talker_config,
        device=device_name,
        dtype=parameter_dtype,
        max_seq_len=2048,
    )
    fast_model = FasterQwen3TTS(
        base_model=qwen3tts,
        predictor_graph=predictor_graph,
        talker_graph=talker_graph,
        device=device_name,
        dtype=parameter_dtype,
        max_seq_len=2048,
    )
    return fast_model, (predictor_graph, talker_graph)


def _dispose_faster_eval_model(fast_model, graph_objects) -> None:
    """Release graph state after one optional monitoring sample."""
    del fast_model
    for graph in graph_objects or ():
        # The graph objects reference the live model modules. Clearing their
        # capture/cache state makes the temporary evaluation wrapper collectable
        # before the next optimizer epoch continues.
        for name in ("graph", "static_cache", "attn_mask", "attn_mask_table", "prefill_attn", "decode_attn"):
            if hasattr(graph, name):
                setattr(graph, name, None)
    # Avoid torch.cuda.empty_cache() here. It can force a device-wide
    # synchronization immediately after generation while the training worker
    # is waiting on the joined evaluation thread. The graph references above
    # are cleared; PyTorch's allocator can reuse the blocks without a cache
    # flush.


def run_prompted_evaluation(
    *,
    qwen3tts,
    prepared_model,
    accelerator,
    record: dict[str, Any],
    eval_text: str,
    step: int,
    output_dir: str | Path,
    writer,
    seed: int,
    use_fast: bool = False,
    tensorboard_prefix: str = "eval",
    audio_prefix: str = "eval",
) -> dict[str, Any]:
    """Generate one deterministic voice-clone sample from the current model.

    Voice-clone conditioning is supported by the 12 Hz Base model for both a
    full SFT model and a PEFT-wrapped model. Passing a string path explicitly
    avoids the WindowsPath input mismatch in the inference wrapper.
    """
    import numpy as np
    import torch
    import soundfile as sf

    reference_audio = str(record.get("ref_audio") or record.get("audio") or "")
    reference_text = str(record.get("text") or "").strip()
    language = str(record.get("language") or "Auto")
    if not reference_audio or not Path(reference_audio).is_file():
        raise FileNotFoundError(f"Evaluation reference audio not found: {reference_audio}")
    if not reference_text:
        raise ValueError("Evaluation reference transcript is empty.")

    original_model = qwen3tts.model
    model = accelerator.unwrap_model(prepared_model)
    was_training = bool(model.training)
    original_device = getattr(qwen3tts, "device", None)
    qwen3tts.model = model
    model.eval()
    started = time.perf_counter()
    heartbeat_stop = threading.Event()
    heartbeat_thread = None
    fast_model = None
    fast_graphs = None
    try:
        def evaluation_heartbeat():
            while not heartbeat_stop.wait(5.0):
                elapsed = time.perf_counter() - started
                print(
                    f"[EVAL] step={int(step)} still generating · elapsed={elapsed:.1f}s",
                    flush=True,
                )

        heartbeat_thread = threading.Thread(target=evaluation_heartbeat, daemon=True, name="qwen-eval-heartbeat")
        heartbeat_thread.start()
        # Accelerate may return a wrapped/PEFT model whose device is correct
        # while the inference wrapper still keeps the device captured when
        # the base model was loaded.  Qwen's tokenizer helper uses that
        # wrapper attribute for input_ids, so synchronize it explicitly.
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        qwen3tts.device = model_device

        def move_tensors(value):
            if torch.is_tensor(value):
                return value.to(model_device)
            if isinstance(value, list):
                return [move_tensors(item) for item in value]
            if isinstance(value, tuple):
                return tuple(move_tensors(item) for item in value)
            if isinstance(value, dict):
                return {key: move_tensors(item) for key, item in value.items()}
            return value

        # Build the prompt before generation and move both codec codes and
        # speaker embeddings.  The audio tokenizer can legitimately leave
        # these tensors on CPU even when the talker is on CUDA; passing that
        # mixed prompt into index_select produces the exact error reported by
        # the Evaluation Zone.
        prompt_items = qwen3tts.create_voice_clone_prompt(
            ref_audio=reference_audio,
            ref_text=reference_text,
            x_vector_only_mode=False,
        )
        if not prompt_items or any(
            item.x_vector_only_mode or not item.icl_mode or item.ref_code is None
            for item in prompt_items
        ):
            raise RuntimeError(
                "Evaluation conditioning did not resolve to ICL. "
                "The evaluation reference must include audio, transcript and codec codes."
            )
        for item in prompt_items:
            item.ref_code = move_tensors(item.ref_code)
            item.ref_spk_embedding = move_tensors(item.ref_spk_embedding)

        print(
            f"[EVAL] step={int(step)} preparing ICL prompted generation "
            f"(max_new_tokens={EVAL_MAX_NEW_TOKENS}, faster_qwen_requested={bool(use_fast)})",
            flush=True,
        )
        # Evaluation is sequential in the training worker. Preserve the
        # optimizer RNG stream so the monitoring sample does not alter the
        # subsequent training updates.
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            torch.manual_seed(int(seed) + int(step))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed) + int(step))
            with torch.inference_mode():
                engine = "Standard Qwen"
                if use_fast:
                    try:
                        fast_model, fast_graphs = _build_faster_eval_model(
                            qwen3tts=qwen3tts,
                            model=model,
                            model_device=model_device,
                        )
                        print(
                            f"[EVAL] step={int(step)} Faster Qwen CUDA Graphs selected; "
                            "capturing/reusing graphs in the training worker",
                            flush=True,
                        )
                    except Exception as exc:
                        print(
                            f"[EVAL] step={int(step)} Faster Qwen unavailable; "
                            f"falling back to Standard Qwen: {exc}",
                            flush=True,
                        )
                        fast_model = None
                        fast_graphs = None

                try:
                    if fast_model is not None:
                        wavs, sample_rate = fast_model.generate_voice_clone(
                            text=str(eval_text or DEFAULT_EVAL_TEXT),
                            language=language,
                            ref_text=reference_text,
                            voice_clone_prompt=prompt_items,
                            non_streaming_mode=True,
                            do_sample=False,
                            max_new_tokens=EVAL_MAX_NEW_TOKENS,
                        )
                        engine = "Faster Qwen CUDA Graphs"
                    else:
                        wavs, sample_rate = qwen3tts.generate_voice_clone(
                            text=str(eval_text or DEFAULT_EVAL_TEXT),
                            language=language,
                            voice_clone_prompt=prompt_items,
                            non_streaming_mode=True,
                            do_sample=False,
                            max_new_tokens=EVAL_MAX_NEW_TOKENS,
                        )
                except Exception as faster_exc:
                    if fast_model is None:
                        raise
                    print(
                        f"[EVAL] step={int(step)} Faster Qwen generation failed; "
                        f"falling back to Standard Qwen: {faster_exc}",
                        flush=True,
                    )
                    _dispose_faster_eval_model(fast_model, fast_graphs)
                    fast_model = None
                    fast_graphs = None
                    torch.set_rng_state(cpu_rng_state)
                    if cuda_rng_state is not None:
                        torch.cuda.set_rng_state_all(cuda_rng_state)
                    torch.manual_seed(int(seed) + int(step))
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(int(seed) + int(step))
                    try:
                        wavs, sample_rate = qwen3tts.generate_voice_clone(
                            text=str(eval_text or DEFAULT_EVAL_TEXT),
                            language=language,
                            voice_clone_prompt=prompt_items,
                            non_streaming_mode=True,
                            do_sample=False,
                            max_new_tokens=EVAL_MAX_NEW_TOKENS,
                        )
                    except Exception as standard_exc:
                        raise RuntimeError(
                            "Faster Qwen and Standard Qwen ICL generation both failed. "
                            f"Faster: {faster_exc}; Standard: {standard_exc}"
                        ) from standard_exc
        finally:
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
        if fast_model is not None:
            _dispose_faster_eval_model(fast_model, fast_graphs)
            fast_model = None
            fast_graphs = None
        if not wavs:
            raise RuntimeError("Qwen returned no evaluation audio.")
        audio = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        audio = np.clip(audio, -1.0, 1.0)
        elapsed = time.perf_counter() - started
        sample_rate = int(sample_rate)
        audio_dir = Path(output_dir) / "evaluation_audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path = audio_dir / f"{str(audio_prefix or 'eval')}_step_{int(step):08d}.wav"
        sf.write(str(audio_path), audio, sample_rate, subtype="PCM_16")
        if writer is not None:
            prefix = str(tensorboard_prefix or "eval").rstrip("/")
            writer.add_audio(f"{prefix}/generated_audio", torch.from_numpy(audio).unsqueeze(0), int(step), sample_rate=sample_rate)
            writer.add_scalar(f"{prefix}/generated_seconds", float(len(audio) / max(1, sample_rate)), int(step))
            writer.add_scalar(f"{prefix}/inference_seconds", float(elapsed), int(step))
            writer.add_text(f"{prefix}/engine", str(engine), int(step))
            writer.add_text(f"{prefix}/conditioning", "ICL", int(step))
            # SummaryWriter owns an asynchronous event writer. Its normal
            # flush cadence and final close() persist the queued sample; a
            # synchronous flush here can hold the evaluation thread after the
            # inference has already completed.
        return {
            "step": int(step),
            "audio_path": str(audio_path),
            "seconds": float(len(audio) / max(1, sample_rate)),
            "elapsed": float(elapsed),
            "reference_audio": reference_audio,
            "conditioning_mode": "ICL",
            "engine": engine,
        }
    finally:
        heartbeat_stop.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=1.0)
        qwen3tts.model = original_model
        qwen3tts.device = original_device
        if fast_model is not None:
            _dispose_faster_eval_model(fast_model, fast_graphs)
        model.train(was_training)


def run_custom_voice_evaluation(
    *,
    qwen3tts,
    prepared_model,
    accelerator,
    speaker_name: str,
    speaker_embedding,
    eval_text: str,
    step: int,
    output_dir: str | Path,
    writer,
    seed: int,
    use_fast: bool = False,
) -> dict[str, Any]:
    """Evaluate a full-SFT run through its published CustomVoice route.

    Full Qwen SFT is trained from the Base checkpoint, but its published
    deployment artifact is converted to ``custom_voice`` and receives the
    trained speaker embedding at codec row 3000.  Temporarily applying those
    exact deployment semantics to the live worker model makes the monitoring
    sample representative of the artifact users will actually select.  The
    original Base configuration is restored before optimization resumes.
    """
    import numpy as np
    import torch
    import soundfile as sf

    speaker_name = str(speaker_name or "speaker_1").strip() or "speaker_1"
    # Qwen's low-level generation path always looks up speaker IDs with
    # ``speaker.lower()``. Keep the display name untouched for logs, but use
    # the canonical lowercase key in the temporary deployment registry.
    speaker_id = speaker_name.lower()
    if speaker_embedding is None:
        raise RuntimeError("CustomVoice evaluation requires the trained speaker embedding.")

    original_model = qwen3tts.model
    model = accelerator.unwrap_model(prepared_model)
    was_training = bool(model.training)
    original_device = getattr(qwen3tts, "device", None)
    model_config = getattr(model, "config", None)
    talker_config = getattr(model_config, "talker_config", None)
    if model_config is None or talker_config is None:
        raise RuntimeError("The current training model has no Qwen talker configuration.")
    speaker_ids = getattr(talker_config, "spk_id", None)
    speaker_dialects = getattr(talker_config, "spk_is_dialect", None)
    if not isinstance(speaker_ids, dict) or not isinstance(speaker_dialects, dict):
        raise RuntimeError("The current Qwen model has no mutable speaker registry.")
    codec_embedding = getattr(getattr(model, "talker", None), "model", None)
    codec_embedding = getattr(codec_embedding, "codec_embedding", None)
    if codec_embedding is None or not hasattr(codec_embedding, "weight"):
        raise RuntimeError("The current Qwen model has no codec speaker embedding table.")
    target_row = 3000
    if int(codec_embedding.weight.shape[0]) <= target_row:
        raise RuntimeError("The Qwen codec embedding table has no trained speaker row 3000.")

    original_tts_type = getattr(model, "tts_model_type", None)
    original_config_type = getattr(model_config, "tts_model_type", None)
    original_speaker_ids = dict(speaker_ids)
    original_speaker_dialects = dict(speaker_dialects)
    original_supported_speakers = getattr(model, "supported_speakers", None)
    original_row = codec_embedding.weight[target_row].detach().clone()
    fast_model = None
    fast_graphs = None
    heartbeat_stop = threading.Event()
    heartbeat_thread = None
    started = time.perf_counter()
    stage = {"name": "initializing"}

    def _set_stage(name: str) -> None:
        stage["name"] = str(name)
        print(f"[EVAL] step={int(step)} stage={stage['name']}", flush=True)

    def _set_deployment_registry() -> None:
        model.tts_model_type = "custom_voice"
        model_config.tts_model_type = "custom_voice"
        speaker_ids.clear()
        speaker_ids[speaker_id] = target_row
        speaker_dialects.clear()
        speaker_dialects[speaker_id] = False
        # The official model stores a dict_keys view here. Reassigning it is
        # safe for model variants that materialize the view at construction.
        model.supported_speakers = speaker_ids.keys()
        value = speaker_embedding[0] if getattr(speaker_embedding, "ndim", 0) > 1 else speaker_embedding
        value = value.detach().to(device=codec_embedding.weight.device, dtype=codec_embedding.weight.dtype)
        with torch.no_grad():
            codec_embedding.weight[target_row].copy_(value)

    def _restore_base_registry() -> None:
        model.tts_model_type = original_tts_type
        model_config.tts_model_type = original_config_type
        speaker_ids.clear()
        speaker_ids.update(original_speaker_ids)
        speaker_dialects.clear()
        speaker_dialects.update(original_speaker_dialects)
        if original_supported_speakers is not None:
            model.supported_speakers = original_supported_speakers
        else:
            model.supported_speakers = speaker_ids.keys()
        with torch.no_grad():
            codec_embedding.weight[target_row].copy_(original_row)

    qwen3tts.model = model
    model.eval()
    try:
        _set_deployment_registry()
    except Exception:
        _restore_base_registry()
        qwen3tts.model = original_model
        qwen3tts.device = original_device
        model.train(was_training)
        raise
    try:
        def evaluation_heartbeat():
            while not heartbeat_stop.wait(5.0):
                elapsed = time.perf_counter() - started
                print(
                    f"[EVAL] step={int(step)} still active · stage={stage['name']} · elapsed={elapsed:.1f}s",
                    flush=True,
                )

        heartbeat_thread = threading.Thread(target=evaluation_heartbeat, daemon=True, name="qwen-eval-heartbeat")
        heartbeat_thread.start()
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _set_stage("preparing CustomVoice generation")
        qwen3tts.device = model_device
        print(
            f"[EVAL] step={int(step)} preparing deployment-matched CustomVoice generation "
            f"speaker={speaker_name} (max_new_tokens={EVAL_MAX_NEW_TOKENS}, "
            f"faster_qwen_requested={bool(use_fast)})",
            flush=True,
        )
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            torch.manual_seed(int(seed) + int(step))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed) + int(step))
            with torch.inference_mode():
                engine = "Standard Qwen"
                if use_fast:
                    _set_stage("building Faster Qwen CUDA Graphs")
                    try:
                        fast_model, fast_graphs = _build_faster_eval_model(
                            qwen3tts=qwen3tts,
                            model=model,
                            model_device=model_device,
                        )
                        print(
                            f"[EVAL] step={int(step)} Faster Qwen CUDA Graphs selected for CustomVoice; "
                            "capturing/reusing graphs in the training worker",
                            flush=True,
                        )
                    except Exception as exc:
                        print(
                            f"[EVAL] step={int(step)} Faster Qwen unavailable for CustomVoice; "
                            f"falling back to Standard Qwen: {exc}",
                            flush=True,
                        )
                        fast_model = None
                        fast_graphs = None

                try:
                    _set_stage("generating audio with Faster Qwen" if fast_model is not None else "generating audio with Standard Qwen")
                    generation_kwargs = {
                        "text": str(eval_text or DEFAULT_EVAL_TEXT),
                        "speaker": speaker_name,
                        "language": "Auto",
                        "instruct": None,
                        "non_streaming_mode": True,
                        "do_sample": False,
                        "max_new_tokens": EVAL_MAX_NEW_TOKENS,
                    }
                    if fast_model is not None:
                        wavs, sample_rate = fast_model.generate_custom_voice(**generation_kwargs)
                        engine = "Faster Qwen CUDA Graphs"
                    else:
                        wavs, sample_rate = qwen3tts.generate_custom_voice(**generation_kwargs)
                    _set_stage("generation returned")
                except Exception as faster_exc:
                    if fast_model is None:
                        raise
                    print(
                        f"[EVAL] step={int(step)} Faster Qwen CustomVoice generation failed; "
                        f"falling back to Standard Qwen: {faster_exc}",
                        flush=True,
                    )
                    _dispose_faster_eval_model(fast_model, fast_graphs)
                    fast_model = None
                    fast_graphs = None
                    torch.set_rng_state(cpu_rng_state)
                    if cuda_rng_state is not None:
                        torch.cuda.set_rng_state_all(cuda_rng_state)
                    torch.manual_seed(int(seed) + int(step))
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(int(seed) + int(step))
                    try:
                        _set_stage("generating audio with Standard Qwen fallback")
                        wavs, sample_rate = qwen3tts.generate_custom_voice(**generation_kwargs)
                        _set_stage("Standard Qwen fallback returned")
                    except Exception as standard_exc:
                        raise RuntimeError(
                            "Faster Qwen and Standard Qwen CustomVoice generation both failed. "
                            f"Faster: {faster_exc}; Standard: {standard_exc}"
                        ) from standard_exc
        finally:
            _set_stage("restoring random state")
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
        if fast_model is not None:
            _set_stage("releasing Faster Qwen CUDA Graphs")
            _dispose_faster_eval_model(fast_model, fast_graphs)
            fast_model = None
            fast_graphs = None
            _set_stage("Faster Qwen CUDA Graphs released")
        if not wavs:
            raise RuntimeError("Qwen returned no evaluation audio.")
        _set_stage("post-processing generated audio")
        audio = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        audio = np.clip(audio, -1.0, 1.0)
        elapsed = time.perf_counter() - started
        sample_rate = int(sample_rate)
        audio_dir = Path(output_dir) / "evaluation_audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path = audio_dir / f"eval_step_{int(step):08d}.wav"
        _set_stage("writing evaluation WAV")
        sf.write(str(audio_path), audio, sample_rate, subtype="PCM_16")
        _set_stage("evaluation WAV written")
        if writer is not None:
            _set_stage("writing TensorBoard audio")
            writer.add_audio("eval/generated_audio", torch.from_numpy(audio).unsqueeze(0), int(step), sample_rate=sample_rate)
            _set_stage("writing TensorBoard scalars")
            writer.add_scalar("eval/generated_seconds", float(len(audio) / max(1, sample_rate)), int(step))
            writer.add_scalar("eval/inference_seconds", float(elapsed), int(step))
            writer.add_text("eval/engine", str(engine), int(step))
            writer.add_text("eval/conditioning", "CustomVoice", int(step))
            _set_stage("TensorBoard event queued")
        _set_stage("returning evaluation result")
        return {
            "step": int(step),
            "audio_path": str(audio_path),
            "seconds": float(len(audio) / max(1, sample_rate)),
            "elapsed": float(elapsed),
            "speaker": speaker_name,
            "conditioning_mode": "CustomVoice",
            "engine": engine,
        }
    finally:
        _set_stage("stopping evaluation heartbeat")
        heartbeat_stop.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=1.0)
        if fast_model is not None:
            _set_stage("final Faster Qwen cleanup")
            _dispose_faster_eval_model(fast_model, fast_graphs)
        _set_stage("restoring training model")
        _restore_base_registry()
        qwen3tts.model = original_model
        qwen3tts.device = original_device
        model.train(was_training)
        _set_stage("evaluation cleanup complete")
