# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
import os
import random
import shutil
import sys
import threading
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from dataset import TTSDataset
from eval_utils import (
    DEFAULT_EVAL_TEXT,
    create_tensorboard_writer,
    load_eval_record,
    log_training_step,
    log_training_epoch,
    place_speech_tokenizer_on_device,
    run_prompted_evaluation,
    run_custom_voice_evaluation,
    write_evaluation_contract,
)
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import load_file, save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig
from training_utils import (
    LR_SCHEDULE_CHOICES,
    build_lr_scheduler,
    checkpoint_completed_epoch,
    load_training_state,
    optimizer_learning_rate,
    restore_scheduler_position,
)

target_speaker_embedding = None
MODEL_EXPORT_MODES = ("Non-ICL-variant", "ICL-variant", "Both")


def _run_evaluation_in_paused_thread(eval_call, accelerator, epoch, step):
    """Run one evaluation on a dedicated thread while training stays paused.

    The training loop must not update weights while an evaluation sample is
    using the same model and CUDA memory.  A joined worker thread preserves
    that sequencing, while avoiding the generation call living on the main
    training thread (which previously made the worker look permanently stuck
    when Qwen generation took a long time).
    """
    result_box = {}
    error_box = {}

    def worker():
        try:
            result_box["result"] = eval_call()
        except BaseException:
            error_box["exc_info"] = sys.exc_info()

    thread = threading.Thread(
        target=worker,
        name=f"qwen-eval-epoch-{int(epoch)}-step-{int(step)}",
        daemon=False,
    )
    accelerator.print(
        f"[EVAL] epoch={int(epoch)} | step={int(step)} | "
        f"starting dedicated evaluation thread '{thread.name}'; training paused"
    )
    started = time.perf_counter()
    thread.start()
    while thread.is_alive():
        # The one-second join interval keeps the main worker responsive enough
        # to emit a heartbeat without allowing training to resume early.
        thread.join(timeout=1.0)
        elapsed = time.perf_counter() - started
        if thread.is_alive() and int(elapsed) > 0 and int(elapsed) % 10 == 0:
            accelerator.print(
                f"[EVAL] epoch={int(epoch)} | step={int(step)} | "
                f"dedicated evaluation thread still running; training paused · elapsed={elapsed:.1f}s"
            )

    if "exc_info" in error_box:
        _, exc, traceback = error_box["exc_info"]
        raise exc.with_traceback(traceback)
    if "result" not in result_box:
        raise RuntimeError("Evaluation thread ended without a result.")

    elapsed = time.perf_counter() - started
    accelerator.print(
        f"[EVAL] epoch={int(epoch)} | step={int(step)} | "
        f"dedicated evaluation thread completed; resuming training · elapsed={elapsed:.1f}s"
    )
    return result_box["result"]


def train():
    global target_speaker_embedding

    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lr_schedule", type=str, default="warmup_cosine", choices=list(LR_SCHEDULE_CHOICES))
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--min_lr_ratio", type=float, default=0.20)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--speaker_name", type=str, default="speaker_test")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--attention", type=str, default="auto", choices=["auto", "flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--save_every_epochs", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--eval_jsonl", type=str, default=None)
    parser.add_argument("--eval_every_epochs", type=int, default=0)
    parser.add_argument("--eval_text", type=str, default=DEFAULT_EVAL_TEXT)
    parser.add_argument("--eval_enabled", action="store_true")
    parser.add_argument("--eval_fast", action="store_true")
    parser.add_argument("--eval_icl", action="store_true", help="Also evaluate the current full-SFT model through Base ICL conditioning.")
    parser.add_argument("--model_export_mode", choices=list(MODEL_EXPORT_MODES), default="ICL-variant", help="Compatibility option; current Full SFT exports one hybrid ICL checkpoint.")
    # Accept the former worker flag for old hand-written launch commands.
    parser.add_argument("--publish_icl_variant", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--resume_model_path", type=str, default=None)
    args = parser.parse_args()
    if args.publish_icl_variant and args.model_export_mode == "Non-ICL-variant":
        args.model_export_mode = "Both"
    # Keep the worker's public behavior aligned with the GUI: the canonical
    # full-SFT result is one hybrid Base checkpoint. Legacy flags are accepted
    # for old launchers, but a new run never publishes a separate CustomVoice
    # artifact.
    args.model_export_mode = "ICL-variant"
    args.num_epochs = max(1, int(args.num_epochs))
    resume_completed_epoch = checkpoint_completed_epoch(args.resume_model_path)
    resume_metadata = load_training_state(args.resume_model_path)
    if args.resume_model_path and resume_completed_epoch >= args.num_epochs:
        raise ValueError(
            f"Selected checkpoint already contains {resume_completed_epoch} completed epoch(s), "
            f"but Epochs is set to {args.num_epochs}. Set Epochs above {resume_completed_epoch} "
            "to continue, or choose Fresh / None."
        )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    mixed_precision = "bf16" if torch.cuda.is_available() else "no"
    accelerator = Accelerator(gradient_accumulation_steps=max(1, args.gradient_accumulation_steps), mixed_precision=mixed_precision)

    MODEL_PATH = args.init_model_path

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    # FlashAttention 2 remains available as an explicit opt-in.  In training
    # Auto starts with SDPA because the Qwen issue tracker documents NaN
    # logits with the FlashAttention backend on some environments.
    attention_chain = [args.attention] if args.attention != "auto" else ["sdpa", "eager"]
    last_error = None
    for attention in attention_chain:
        try:
            load_kwargs = {"dtype": dtype, "attn_implementation": attention}
            if args.device:
                load_kwargs["device_map"] = args.device
            qwen3tts = Qwen3TTSModel.from_pretrained(MODEL_PATH, **load_kwargs)
            accelerator.print(f"Loaded Qwen3-TTS with attention={attention}")
            break
        except Exception as exc:
            last_error = exc
            if "dtype" in str(exc) and "torch_dtype" in str(exc):
                try:
                    fallback_kwargs = {"torch_dtype": dtype, "attn_implementation": attention}
                    if args.device:
                        fallback_kwargs["device_map"] = args.device
                    qwen3tts = Qwen3TTSModel.from_pretrained(MODEL_PATH, **fallback_kwargs)
                    accelerator.print(f"Loaded Qwen3-TTS with attention={attention}")
                    break
                except Exception as fallback_exc:
                    last_error = fallback_exc
            accelerator.print(f"Attention backend {attention} unavailable: {exc}")
    else:
        raise RuntimeError(f"Could not load Qwen3-TTS with {attention_chain}: {last_error}")
    if args.resume_model_path:
        resume_weights = os.path.join(args.resume_model_path, "model.safetensors")
        if not os.path.isfile(resume_weights):
            raise FileNotFoundError(f"Resume checkpoint is missing model.safetensors: {args.resume_model_path}")
        resume_state = load_file(resume_weights, device="cpu")
        missing, unexpected = qwen3tts.model.load_state_dict(resume_state, strict=False)
        accelerator.print(
            f"Resumed full SFT weights from {args.resume_model_path} · "
            f"completed_epoch={resume_completed_epoch} · missing={len(missing)} unexpected={len(unexpected)}"
        )
    config = AutoConfig.from_pretrained(MODEL_PATH)

    train_data = open(args.train_jsonl).readlines()
    train_data = [json.loads(line) for line in train_data]
    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    steps_per_epoch = max(
        1,
        (len(train_dataloader) + max(1, args.gradient_accumulation_steps) - 1)
        // max(1, args.gradient_accumulation_steps),
    )
    total_optimizer_steps = steps_per_epoch * args.num_epochs
    resume_global_step = resume_completed_epoch * steps_per_epoch
    accelerator.print(
        f"[TRAIN] Schedule | steps_per_epoch={steps_per_epoch} | "
        f"total_steps={total_optimizer_steps} | epochs={args.num_epochs}"
    )
    if args.resume_model_path:
        accelerator.print(
            f"[TRAIN] Resume position | starting_epoch={resume_completed_epoch + 1} | "
            f"completed_epochs={resume_completed_epoch} | global_step={resume_global_step} | "
            "optimizer moments are reset because this checkpoint stores model weights only"
        )
    accelerator.print(
        "[TRAIN] Full-SFT export | hybrid ICL checkpoint | "
        "ICL=enabled | speaker_only=enabled"
    )

    eval_record, eval_source = load_eval_record(args.eval_jsonl, args.train_jsonl)
    evaluation_enabled = bool(args.eval_enabled and eval_record and int(args.eval_every_epochs) > 0)
    if args.eval_enabled and not eval_record:
        accelerator.print("[EVAL] Evaluation Zone requested, but no usable reference record was found; audio evaluation is disabled.")
    writer = None
    if accelerator.is_main_process:
        write_evaluation_contract(
            args.output_model_path,
            enabled=evaluation_enabled,
            text=args.eval_text,
            every_epochs=args.eval_every_epochs,
            source=eval_source,
            record=eval_record,
            fast_requested=args.eval_fast,
            conditioning_mode="Hybrid Base ICL" if args.eval_icl else "CustomVoice",
            icl_enabled=args.eval_icl,
        )
        writer = create_tensorboard_writer(
            args.output_model_path,
            {
                "training_type": "qwen_compatible_full_sft",
                "base_model": MODEL_PATH,
                "eval_enabled": evaluation_enabled,
                "eval_every_epochs": max(0, int(args.eval_every_epochs)),
                "eval_text": str(args.eval_text or DEFAULT_EVAL_TEXT),
                "eval_source": eval_source,
                "eval_conditioning_mode": "Hybrid Base ICL" if args.eval_icl else "CustomVoice",
                "eval_icl_enabled": bool(args.eval_icl),
                "faster_qwen_requested": bool(args.eval_fast),
                "learning_rate": float(args.lr),
                "lr_schedule": args.lr_schedule,
                "warmup_ratio": float(args.warmup_ratio),
                "min_lr_ratio": float(args.min_lr_ratio),
                "weight_decay": float(args.weight_decay),
            },
        )
        if evaluation_enabled:
            accelerator.print(
                f"[EVAL] Fixed reference: {eval_source} · every {int(args.eval_every_epochs)} epoch(s) · "
                f"Hybrid ICL={'yes' if args.eval_icl else 'no'}"
            )

    optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=max(0.0, args.weight_decay))

    model, optimizer, train_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader
    )
    scheduler, warmup_steps, min_lr_ratio = build_lr_scheduler(
        optimizer,
        total_steps=total_optimizer_steps,
        schedule=args.lr_schedule,
        warmup_ratio=args.warmup_ratio,
        min_lr_ratio=args.min_lr_ratio,
    )
    restore_scheduler_position(scheduler, optimizer, resume_global_step)
    accelerator.print(
        f"[TRAIN] LR schedule | mode={args.lr_schedule} | peak={args.lr:.3e} | "
        f"floor={args.lr * min_lr_ratio:.3e} ({min_lr_ratio:.0%}) | warmup_steps={warmup_steps}"
    )
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = "unknown"
    accelerator.print(
        f"[TRAIN] accelerator_device={accelerator.device} · model_device={model_device} · "
        f"cuda_available={torch.cuda.is_available()}"
    )
    place_speech_tokenizer_on_device(qwen3tts, model_device, accelerator.print)

    num_epochs = args.num_epochs
    model.train()

    def write_packaged_speaker_prompt(output_dir):
        """Bundle the learned speaker x-vector for reference-free inference."""
        if target_speaker_embedding is None:
            return
        embedding = target_speaker_embedding.detach().to("cpu")
        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)
        embedding = embedding[:1].contiguous()
        save_file({"ref_spk_embedding": embedding}, os.path.join(output_dir, "trained_speaker_embedding.safetensors"))
        Path(output_dir, "trained_speaker_prompt.json").write_text(
            json.dumps(
                {
                    "format": "qwen3_tts_voice_clone_prompt",
                    "conditioning": "speaker-only",
                    "reference_free": True,
                    "supports_icl": True,
                    "speaker_name": args.speaker_name,
                    "embedding_file": "trained_speaker_embedding.safetensors",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def write_packaged_icl_prompt(output_dir):
        """Bundle one prepared dataset reference for audio-free ICL inference."""
        if target_speaker_embedding is None or not eval_record:
            return
        reference_text = str(eval_record.get("text") or "").strip()
        raw_codes = eval_record.get("audio_codes")
        if not reference_text or not isinstance(raw_codes, list) or not raw_codes:
            accelerator.print("[TRAIN] Packaged ICL prompt skipped: the selected reference has no transcript or prepared audio_codes.")
            return
        try:
            ref_code = torch.tensor(raw_codes, dtype=torch.long)
            if ref_code.ndim != 2 or ref_code.shape[0] < 1 or ref_code.shape[1] < 1:
                raise ValueError(f"invalid audio_codes shape {tuple(ref_code.shape)}")
            embedding = target_speaker_embedding.detach().to("cpu")
            if embedding.ndim == 1:
                embedding = embedding.unsqueeze(0)
            embedding = embedding[:1].contiguous()
            save_file(
                {"ref_spk_embedding": embedding, "ref_code": ref_code.contiguous()},
                os.path.join(output_dir, "trained_icl_prompt.safetensors"),
            )
            Path(output_dir, "trained_icl_prompt.json").write_text(
                json.dumps(
                    {
                        "format": "qwen3_tts_voice_clone_prompt",
                        "conditioning": "ICL",
                        "reference_free": True,
                        "supports_icl": True,
                        "speaker_name": args.speaker_name,
                        "ref_text": reference_text,
                        "reference_language": str(eval_record.get("language") or "Auto"),
                        "source_audio": str(eval_record.get("ref_audio") or eval_record.get("audio") or ""),
                        "prompt_file": "trained_icl_prompt.safetensors",
                        "codec_frames": int(ref_code.shape[0]),
                        "codec_channels": int(ref_code.shape[1]),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            accelerator.print(f"[TRAIN] Packaged reference-free ICL prompt: {output_dir}")
        except Exception as exc:
            accelerator.print(f"[TRAIN] Packaged ICL prompt skipped: {exc}")

    def write_training_state(output_dir, completed_epoch):
        Path(output_dir, "training_state.json").write_text(
            json.dumps(
                {
                    "completed_epoch": int(completed_epoch),
                    "global_step": int(global_step),
                    "steps_per_epoch": int(steps_per_epoch),
                    "total_epochs": int(num_epochs),
                    "loss_ema": float(loss_ema) if loss_ema is not None else None,
                    "optimizer_state": "not_saved_model_only_checkpoint",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def save_checkpoint(checkpoint_name, create_icl_variant=False):
        if not accelerator.is_main_process or target_speaker_embedding is None:
            return
        output_dir = os.path.join(args.output_model_path, checkpoint_name)

        if args.model_export_mode == "ICL-variant":
            # Canonical current path: save one complete Base-compatible
            # checkpoint.  It supports both transcript-conditioned ICL and
            # speaker-only inference; no second CustomVoice copy is emitted.
            shutil.copytree(MODEL_PATH, output_dir, dirs_exist_ok=True)
            with open(os.path.join(output_dir, "config.json"), "r", encoding="utf-8") as f:
                hybrid_config = json.load(f)
            hybrid_config["tts_model_type"] = "base"
            with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump(hybrid_config, f, indent=2, ensure_ascii=False)

            unwrapped_model = accelerator.unwrap_model(model)
            state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}
            save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
            with open(os.path.join(output_dir, "qwen_easy_training.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "speaker_name": args.speaker_name,
                    "base_model": MODEL_PATH,
                    "checkpoint": checkpoint_name,
                    "training_type": "qwen_compatible_full_sft_hybrid_icl",
                    "variant": "base_icl",
                    "supports_icl": True,
                    "supports_speaker_only": True,
                    "reference_free_speaker_prompt": True,
                    "reference_free_icl_prompt": bool(eval_record and eval_record.get("audio_codes") and eval_record.get("text")),
                    "model_export_mode": "ICL-variant",
                    "hybrid_checkpoint": True,
                }, f, indent=2, ensure_ascii=False)
            write_packaged_speaker_prompt(output_dir)
            write_packaged_icl_prompt(output_dir)
            write_training_state(output_dir, completed_epoch)
            accelerator.print(f"[TRAIN] Saved hybrid ICL checkpoint: {output_dir}")
            return

        shutil.copytree(MODEL_PATH, output_dir, dirs_exist_ok=True)

        input_config_file = os.path.join(MODEL_PATH, "config.json")
        output_config_file = os.path.join(output_dir, "config.json")
        with open(input_config_file, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        config_dict["tts_model_type"] = "custom_voice"
        talker_config = config_dict.get("talker_config", {})
        # Qwen's low-level generator normalizes speaker input with .lower()
        # before looking it up. The visible speaker name remains in metadata,
        # while the checkpoint registry uses the canonical key.
        speaker_id = str(args.speaker_name).strip().lower() or "speaker_1"
        talker_config["spk_id"] = {speaker_id: 3000}
        talker_config["spk_is_dialect"] = {speaker_id: False}
        config_dict["talker_config"] = talker_config
        with open(output_config_file, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        unwrapped_model = accelerator.unwrap_model(model)
        state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}
        for key in [k for k in state_dict if k.startswith("speaker_encoder")]:
            del state_dict[key]
        weight = state_dict["talker.model.codec_embedding.weight"]
        state_dict["talker.model.codec_embedding.weight"][3000] = target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
        save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
        with open(os.path.join(output_dir, "qwen_easy_training.json"), "w", encoding="utf-8") as f:
            json.dump({
                "speaker_name": args.speaker_name,
                "base_model": MODEL_PATH,
                "checkpoint": checkpoint_name,
                "training_type": "qwen_compatible_full_sft",
                "variant": "custom_voice",
                "supports_icl": False,
                "model_export_mode": args.model_export_mode,
                "icl_variant_published": args.model_export_mode in {"ICL-variant", "Both"},
            }, f, indent=2)

        if not create_icl_variant:
            return

        # Keep a second artifact with the original Base architecture. Qwen's
        # speaker encoder exists only in Base models, so an ICL-capable output
        # cannot be recovered later from the published CustomVoice checkpoint.
        icl_output_dir = os.path.join(args.output_model_path, "icl_variant")
        shutil.copytree(MODEL_PATH, icl_output_dir, dirs_exist_ok=True)
        icl_config_file = os.path.join(icl_output_dir, "config.json")
        with open(icl_config_file, "r", encoding="utf-8") as f:
            icl_config = json.load(f)
        icl_config["tts_model_type"] = "base"
        with open(icl_config_file, "w", encoding="utf-8") as f:
            json.dump(icl_config, f, indent=2, ensure_ascii=False)
        icl_state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}
        save_file(icl_state_dict, os.path.join(icl_output_dir, "model.safetensors"))
        with open(os.path.join(icl_output_dir, "qwen_easy_training.json"), "w", encoding="utf-8") as f:
            json.dump({
                "speaker_name": args.speaker_name,
                "base_model": MODEL_PATH,
                "checkpoint": checkpoint_name,
                "source_checkpoint": checkpoint_name,
                "training_type": "qwen_compatible_full_sft_icl_variant",
                "variant": "base_icl",
                "supports_icl": True,
                "model_export_mode": args.model_export_mode,
                "icl_variant_published": True,
            }, f, indent=2)
        accelerator.print(f"[TRAIN] Published ICL-capable Base variant: {icl_output_dir}")

    global_step = resume_global_step
    saved_loss_ema = resume_metadata.get("loss_ema")
    loss_ema = float(saved_loss_ema) if isinstance(saved_loss_ema, (int, float)) else None
    try:
        for epoch in range(resume_completed_epoch, num_epochs):
            epoch_loss_total = 0.0
            epoch_updates = 0
            for step, batch in enumerate(train_dataloader):
                did_update = False
                applied_lr = optimizer_learning_rate(optimizer)
                grad_norm_value = None
                with accelerator.accumulate(model):
                    input_ids = batch['input_ids']
                    codec_ids = batch['codec_ids']
                    ref_mels = batch['ref_mels']
                    text_embedding_mask = batch['text_embedding_mask']
                    codec_embedding_mask = batch['codec_embedding_mask']
                    attention_mask = batch['attention_mask']
                    codec_0_labels = batch['codec_0_labels']
                    codec_mask = batch['codec_mask']

                    speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype)).detach()
                    # Keep a final-batch x-vector for the packaged speaker
                    # prompt. The speaker encoder is trainable in Full SFT,
                    # so retaining the first pre-update batch would export a
                    # stale Base embedding instead of the trained speaker.
                    target_speaker_embedding = speaker_embedding.detach().mean(dim=0, keepdim=True)

                    input_text_ids = input_ids[:, :, 0]
                    input_codec_ids = input_ids[:, :, 1]

                    input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
                    input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                    input_codec_embedding[:, 6, :] = speaker_embedding

                    input_embeddings = input_text_embedding + input_codec_embedding

                    for i in range(1, 16):
                        codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                        codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                        input_embeddings = input_embeddings + codec_i_embedding

                    outputs = model.talker(
                        inputs_embeds=input_embeddings[:, :-1, :],
                        attention_mask=attention_mask[:, :-1],
                        labels=codec_0_labels[:, 1:],
                        output_hidden_states=True
                    )

                    hidden_states = outputs.hidden_states[0][-1]
                    talker_hidden_states = hidden_states[codec_mask[:, :-1]]
                    talker_codec_ids = codec_ids[codec_mask]

                    sub_talker_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)

                    loss = outputs.loss + 0.3 * sub_talker_loss
                    if not bool(torch.isfinite(loss.detach()).all().item()):
                        optimizer.zero_grad(set_to_none=True)
                        accelerator.print(f"Epoch {epoch + 1} | Step {global_step + 1} | Loss: {loss.item()}")
                        raise FloatingPointError(
                            "Non-finite Qwen loss detected before optimizer update. "
                            "Use Auto/SDPA, lower Learning Rate, and resume from the last valid checkpoint."
                        )
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        grad_norm = accelerator.clip_grad_norm_(model.parameters(), max(0.0, args.max_grad_norm))
                        if not bool(torch.isfinite(torch.as_tensor(grad_norm)).all().item()):
                            optimizer.zero_grad(set_to_none=True)
                            accelerator.print(f"Epoch {epoch + 1} | Step {global_step + 1} | Loss: {loss.item():.4f}")
                            raise FloatingPointError(
                                "Non-finite Qwen gradient detected before optimizer update. "
                                "Use Auto/SDPA, lower Learning Rate, and resume from the last valid checkpoint."
                            )
                        grad_norm_value = float(torch.as_tensor(grad_norm).detach().float().item())
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()
                        global_step += 1
                        did_update = True

                if not did_update:
                    continue

                update_loss = float(loss.detach().float().item())
                epoch_loss_total += update_loss
                epoch_updates += 1
                loss_ema = update_loss if loss_ema is None else (0.95 * loss_ema + 0.05 * update_loss)

                if global_step == 1 or global_step % 10 == 0:
                    log_training_step(
                        writer,
                        step=global_step,
                        epoch=epoch + 1,
                        loss=update_loss,
                        loss_ema=loss_ema,
                        learning_rate=applied_lr,
                        grad_norm=grad_norm_value,
                    )
                    accelerator.print(
                        f"Epoch {epoch + 1} | Step {global_step} | Loss: {update_loss:.4f} | "
                        f"EMA: {loss_ema:.4f} | LR: {applied_lr:.3e} | "
                        f"GradNorm: {grad_norm_value:.4f}"
                    )

            completed_epoch = epoch + 1
            epoch_loss_mean = epoch_loss_total / max(1, epoch_updates)
            log_training_epoch(writer, epoch=completed_epoch, loss_mean=epoch_loss_mean, step=global_step)
            accelerator.print(
                f"[TRAIN] Epoch {completed_epoch} complete | mean_loss={epoch_loss_mean:.4f} | "
                f"lr_next={optimizer_learning_rate(optimizer):.3e}"
            )
            is_final_epoch = completed_epoch == num_epochs
            should_evaluate = evaluation_enabled and (
                completed_epoch % int(args.eval_every_epochs) == 0 or is_final_epoch
            )
            should_save = (
                (args.save_every_epochs and completed_epoch % int(args.save_every_epochs) == 0)
                or is_final_epoch
            )
            if should_save:
                checkpoint_name = f"checkpoint-epoch-{completed_epoch}"
                save_checkpoint(
                    checkpoint_name,
                    create_icl_variant=is_final_epoch and args.model_export_mode in {"ICL-variant", "Both"},
                )
            if should_evaluate and accelerator.is_main_process:
                try:
                    def _evaluate_current_checkpoint():
                        results = {}
                        if args.eval_icl:
                            try:
                                results["icl"] = run_prompted_evaluation(
                                    qwen3tts=qwen3tts,
                                    prepared_model=model,
                                    accelerator=accelerator,
                                    record=eval_record,
                                    eval_text=args.eval_text,
                                    step=global_step,
                                    output_dir=args.output_model_path,
                                    writer=writer,
                                    seed=args.seed,
                                    use_fast=args.eval_fast,
                                    tensorboard_prefix="eval/icl",
                                    audio_prefix="eval_icl",
                                )
                            except Exception as exc:
                                accelerator.print(
                                    f"[EVAL] epoch={completed_epoch} step={global_step} ICL failed: {exc}"
                                )
                        else:
                            # Compatibility path for an old hand-written
                            # worker command that explicitly omits --eval_icl.
                            try:
                                results["custom_voice"] = run_custom_voice_evaluation(
                                    qwen3tts=qwen3tts,
                                    prepared_model=model,
                                    accelerator=accelerator,
                                    speaker_name=args.speaker_name,
                                    speaker_embedding=target_speaker_embedding,
                                    eval_text=args.eval_text,
                                    step=global_step,
                                    output_dir=args.output_model_path,
                                    writer=writer,
                                    seed=args.seed,
                                    use_fast=args.eval_fast,
                                )
                            except Exception as exc:
                                accelerator.print(
                                    f"[EVAL] epoch={completed_epoch} step={global_step} CustomVoice failed: {exc}"
                                )
                        if not results:
                            raise RuntimeError("Hybrid ICL evaluation failed.")
                        return results

                    result = _run_evaluation_in_paused_thread(
                        _evaluate_current_checkpoint,
                        accelerator,
                        completed_epoch,
                        global_step,
                    )
                    for evaluation_result in result.values():
                        accelerator.print(
                            f"[EVAL] epoch={completed_epoch} | step={global_step} | generated={evaluation_result['seconds']:.2f}s | "
                            f"elapsed={evaluation_result['elapsed']:.2f}s | conditioning={evaluation_result['conditioning_mode']} | "
                            f"engine={evaluation_result.get('engine', 'Standard Qwen')} | "
                            f"audio={evaluation_result['audio_path']}"
                        )
                except Exception as exc:
                    accelerator.print(
                        f"[EVAL] epoch={completed_epoch} step={global_step} skipped; "
                        "evaluation thread ended with an error; training will resume without an evaluation sample: "
                        f"{exc}"
                    )
    finally:
        if writer is not None:
            writer.close()

if __name__ == "__main__":
    train()
