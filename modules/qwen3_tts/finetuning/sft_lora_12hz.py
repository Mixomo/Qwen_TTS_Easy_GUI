"""Experimental PEFT/LoRA fine-tuning for the Qwen3-TTS 12 Hz Base model.

The official Qwen workflow is full SFT. This companion worker keeps the same
dataset and forward path but wraps the conditional model with PEFT and writes
an adapter directory. It is intentionally opt-in and is not used by the
official checkpoint path.
"""
from __future__ import annotations

import argparse
import json
import random
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
    write_evaluation_contract,
)
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from safetensors.torch import save_file
from qwen_tts import Qwen3TTSModel
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", required=True)
    parser.add_argument("--output_model_path", required=True)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lr_schedule", type=str, default="warmup_cosine", choices=list(LR_SCHEDULE_CHOICES))
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--min_lr_ratio", type=float, default=0.20)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--speaker_name", default="speaker_test")
    parser.add_argument("--device", default=None)
    parser.add_argument("--attention", default="auto", choices=["auto", "flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
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
    parser.add_argument("--eval_icl", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--resume_adapter_path", type=str, default=None)
    args = parser.parse_args()
    args.num_epochs = max(1, int(args.num_epochs))
    resume_completed_epoch = checkpoint_completed_epoch(args.resume_adapter_path)
    resume_metadata = load_training_state(args.resume_adapter_path)
    if args.resume_adapter_path and resume_completed_epoch >= args.num_epochs:
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
    accelerator = Accelerator(
        gradient_accumulation_steps=max(1, args.gradient_accumulation_steps),
        mixed_precision=mixed_precision,
    )
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    # FlashAttention 2 remains available as an explicit opt-in.  In training
    # Auto starts with SDPA because the Qwen issue tracker documents NaN
    # logits with the FlashAttention backend on some environments.
    attention_chain = [args.attention] if args.attention != "auto" else ["sdpa", "eager"]
    last_error = None
    for attention in attention_chain:
        try:
            kwargs = {"dtype": dtype, "attn_implementation": attention}
            if args.device:
                kwargs["device_map"] = args.device
            qwen3tts = Qwen3TTSModel.from_pretrained(args.init_model_path, **kwargs)
            accelerator.print(f"Loaded Qwen3-TTS with attention={attention}")
            break
        except Exception as exc:
            last_error = exc
            if "dtype" in str(exc) and "torch_dtype" in str(exc):
                try:
                    fallback_kwargs = {"torch_dtype": dtype, "attn_implementation": attention}
                    if args.device:
                        fallback_kwargs["device_map"] = args.device
                    qwen3tts = Qwen3TTSModel.from_pretrained(args.init_model_path, **fallback_kwargs)
                    accelerator.print(f"Loaded Qwen3-TTS with attention={attention}")
                    break
                except Exception as fallback_exc:
                    last_error = fallback_exc
            accelerator.print(f"Attention backend {attention} unavailable: {exc}")
    else:
        raise RuntimeError(f"Could not load Qwen3-TTS with {attention_chain}: {last_error}")

    config = AutoConfig.from_pretrained(args.init_model_path)
    records = [json.loads(line) for line in Path(args.train_jsonl).read_text(encoding="utf-8").splitlines() if line.strip()]
    dataset = TTSDataset(records, qwen3tts.processor, config)
    loader = DataLoader(dataset, batch_size=max(1, args.batch_size), shuffle=True, collate_fn=dataset.collate_fn)
    steps_per_epoch = max(
        1,
        (len(loader) + max(1, args.gradient_accumulation_steps) - 1)
        // max(1, args.gradient_accumulation_steps),
    )
    total_optimizer_steps = steps_per_epoch * args.num_epochs
    resume_global_step = resume_completed_epoch * steps_per_epoch
    accelerator.print(
        f"[TRAIN] Schedule | steps_per_epoch={steps_per_epoch} | "
        f"total_steps={total_optimizer_steps} | epochs={args.num_epochs}"
    )
    if args.resume_adapter_path:
        accelerator.print(
            f"[TRAIN] Resume position | starting_epoch={resume_completed_epoch + 1} | "
            f"completed_epochs={resume_completed_epoch} | global_step={resume_global_step} | "
            "optimizer moments are reset because this checkpoint stores adapter weights only"
        )

    eval_record, eval_source = load_eval_record(args.eval_jsonl, args.train_jsonl)
    evaluation_enabled = bool(args.eval_enabled and eval_record and int(args.eval_every_epochs) > 0)
    if args.eval_enabled and not eval_record:
        accelerator.print("[EVAL] Evaluation Zone requested, but no usable reference record was found; audio evaluation is disabled.")

    if args.resume_adapter_path:
        model = PeftModel.from_pretrained(qwen3tts.model, str(Path(args.resume_adapter_path)), is_trainable=True)
        accelerator.print(f"Resumed LoRA adapter from {args.resume_adapter_path}")
    else:
        lora_config = LoraConfig(
            r=max(1, args.rank),
            lora_alpha=max(1, args.alpha),
            lora_dropout=max(0.0, args.dropout),
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(qwen3tts.model, lora_config)
    model.print_trainable_parameters()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=max(0.0, args.weight_decay))
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
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
    model.train()
    target_speaker_embedding = None

    def write_packaged_speaker_prompt(output_dir):
        """Bundle the learned speaker x-vector beside the adapter."""
        if target_speaker_embedding is None:
            return
        embedding = target_speaker_embedding.detach().to("cpu")
        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)
        embedding = embedding[:1].contiguous()
        save_file({"ref_spk_embedding": embedding}, str(Path(output_dir) / "trained_speaker_embedding.safetensors"))
        (Path(output_dir) / "trained_speaker_prompt.json").write_text(
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
                str(Path(output_dir) / "trained_icl_prompt.safetensors"),
            )
            (Path(output_dir) / "trained_icl_prompt.json").write_text(
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
            conditioning_mode="ICL",
            icl_enabled=evaluation_enabled,
        )
        writer = create_tensorboard_writer(
            args.output_model_path,
            {
                "training_type": "peft_lora",
                "base_model": args.init_model_path,
                "rank": int(args.rank),
                "alpha": int(args.alpha),
                "dropout": float(args.dropout),
                "eval_enabled": evaluation_enabled,
                "eval_conditioning_mode": "Base ICL",
                "eval_icl_enabled": evaluation_enabled,
                "eval_every_epochs": max(0, int(args.eval_every_epochs)),
                "eval_text": str(args.eval_text or DEFAULT_EVAL_TEXT),
                "eval_source": eval_source,
                "faster_qwen_requested": bool(args.eval_fast),
                "learning_rate": float(args.lr),
                "lr_schedule": args.lr_schedule,
                "warmup_ratio": float(args.warmup_ratio),
                "min_lr_ratio": float(args.min_lr_ratio),
                "weight_decay": float(args.weight_decay),
            },
        )
        if evaluation_enabled:
            accelerator.print(f"[EVAL] Fixed Base ICL reference: {eval_source} · every {int(args.eval_every_epochs)} epoch(s)")

    global_step = resume_global_step
    saved_loss_ema = resume_metadata.get("loss_ema")
    loss_ema = float(saved_loss_ema) if isinstance(saved_loss_ema, (int, float)) else None
    try:
        for epoch in range(resume_completed_epoch, args.num_epochs):
            epoch_loss_total = 0.0
            epoch_updates = 0
            for step, batch in enumerate(loader):
                did_update = False
                applied_lr = optimizer_learning_rate(optimizer)
                grad_norm_value = None
                with accelerator.accumulate(model):
                    speaker_embedding = model.speaker_encoder(batch["ref_mels"].to(model.device).to(model.dtype)).detach()
                    # Keep the current speaker encoder output so each saved
                    # adapter/checkpoint bundles the trained speaker prompt.
                    target_speaker_embedding = speaker_embedding.detach().mean(dim=0, keepdim=True)
                    input_text_ids = batch["input_ids"][:, :, 0]
                    input_codec_ids = batch["input_ids"][:, :, 1]
                    text_embedding = model.talker.model.text_embedding(input_text_ids) * batch["text_embedding_mask"]
                    codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * batch["codec_embedding_mask"]
                    codec_embedding[:, 6, :] = speaker_embedding
                    embeddings = text_embedding + codec_embedding
                    for index in range(1, 16):
                        extra = model.talker.code_predictor.get_input_embeddings()[index - 1](batch["codec_ids"][:, :, index])
                        embeddings = embeddings + extra * batch["codec_mask"].unsqueeze(-1)
                    outputs = model.talker(
                        inputs_embeds=embeddings[:, :-1, :],
                        attention_mask=batch["attention_mask"][:, :-1],
                        labels=batch["codec_0_labels"][:, 1:],
                        output_hidden_states=True,
                    )
                    hidden = outputs.hidden_states[0][-1]
                    talker_hidden = hidden[batch["codec_mask"][:, :-1]]
                    talker_codes = batch["codec_ids"][batch["codec_mask"]]
                    _, sub_loss = model.talker.forward_sub_talker_finetune(talker_codes, talker_hidden)
                    loss = outputs.loss + 0.3 * sub_loss
                    if not bool(torch.isfinite(loss.detach()).all().item()):
                        optimizer.zero_grad(set_to_none=True)
                        accelerator.print(f"Epoch {epoch + 1} | Step {global_step + 1} | Loss {loss.item()}")
                        raise FloatingPointError(
                            "Non-finite Qwen loss detected before optimizer update. "
                            "Use Auto/SDPA, lower Learning Rate, and resume from the last valid checkpoint."
                        )
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        grad_norm = accelerator.clip_grad_norm_(model.parameters(), max(0.0, args.max_grad_norm))
                        if not bool(torch.isfinite(torch.as_tensor(grad_norm)).all().item()):
                            optimizer.zero_grad(set_to_none=True)
                            accelerator.print(f"Epoch {epoch + 1} | Step {global_step + 1} | Loss {loss.item():.4f}")
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
            epoch_loss_mean = epoch_loss_total / max(1, epoch_updates)
            log_training_epoch(writer, epoch=epoch + 1, loss_mean=epoch_loss_mean, step=global_step)
            accelerator.print(
                f"[TRAIN] Epoch {epoch + 1} complete | mean_loss={epoch_loss_mean:.4f} | "
                f"lr_next={optimizer_learning_rate(optimizer):.3e}"
            )
            completed_epoch = epoch + 1
            is_final_epoch = completed_epoch == args.num_epochs
            should_evaluate = evaluation_enabled and (
                completed_epoch % int(args.eval_every_epochs) == 0 or is_final_epoch
            )
            should_save = (
                (args.save_every_epochs and completed_epoch % int(args.save_every_epochs) == 0)
                or is_final_epoch
                or should_evaluate
            )
            if should_save and accelerator.is_main_process:
                checkpoint = Path(args.output_model_path) / f"checkpoint-epoch-{completed_epoch}"
                accelerator.unwrap_model(model).save_pretrained(checkpoint)
                write_packaged_speaker_prompt(checkpoint)
                write_packaged_icl_prompt(checkpoint)
                (checkpoint / "training_state.json").write_text(
                    json.dumps(
                        {
                            "completed_epoch": int(completed_epoch),
                            "global_step": int(global_step),
                            "steps_per_epoch": int(steps_per_epoch),
                            "total_epochs": int(args.num_epochs),
                            "loss_ema": float(loss_ema) if loss_ema is not None else None,
                            "optimizer_state": "not_saved_adapter_only_checkpoint",
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            if should_evaluate and accelerator.is_main_process:
                try:
                    accelerator.print(
                        f"[EVAL] epoch={completed_epoch} | step={global_step} | "
                        "starting sequential ICL evaluation; training paused"
                    )
                    result = run_prompted_evaluation(
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
                    )
                    if str(result.get("conditioning_mode", "")).casefold() != "icl":
                        raise RuntimeError(
                            "PEFT Evaluation Zone returned a non-ICL result; "
                            "the monitoring sample must use Base ICL conditioning."
                        )
                    accelerator.print(
                        f"[EVAL] epoch={completed_epoch} | step={global_step} | generated={result['seconds']:.2f}s | "
                        f"elapsed={result['elapsed']:.2f}s | conditioning={result['conditioning_mode']} | "
                        f"engine={result.get('engine', 'Standard Qwen')} | "
                        f"audio={result['audio_path']}"
                    )
                except Exception as exc:
                    accelerator.print(f"[EVAL] epoch={completed_epoch} step={global_step} failed: {exc}")
    finally:
        if writer is not None:
            writer.close()

    if accelerator.is_main_process:
        output = Path(args.output_model_path)
        output.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(output / "adapter")
        write_packaged_speaker_prompt(output / "adapter")
        write_packaged_icl_prompt(output / "adapter")
        (output / "qwen_easy_lora.json").write_text(
            json.dumps({
                "training_type": "peft_lora",
                "base_model": args.init_model_path,
                "speaker_name": args.speaker_name,
                "rank": args.rank,
                "alpha": args.alpha,
                "epochs": args.num_epochs,
                "reference_free_speaker_prompt": True,
                "reference_free_icl_prompt": bool(eval_record and eval_record.get("audio_codes") and eval_record.get("text")),
            }, indent=2),
            encoding="utf-8",
        )
        accelerator.print(f"Saved LoRA adapter to {output / 'adapter'}")


if __name__ == "__main__":
    main()
