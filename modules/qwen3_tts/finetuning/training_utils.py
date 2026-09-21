"""Small, framework-neutral helpers shared by the Qwen training workers."""
from __future__ import annotations

import math
import re
from pathlib import Path

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


LR_SCHEDULE_CHOICES = ("constant", "warmup_cosine")


def build_lr_scheduler(
    optimizer: Optimizer,
    *,
    total_steps: int,
    schedule: str,
    warmup_ratio: float,
    min_lr_ratio: float,
) -> tuple[LambdaLR, int, float]:
    """Build a conservative per-optimizer-step schedule.

    ``warmup_cosine`` starts at 10% of the requested LR, warms up briefly,
    then decays to ``min_lr_ratio`` of the requested LR.  Keeping a non-zero
    floor avoids turning the last epochs into a frozen checkpoint.  The
    constant option remains available for compatibility and comparisons.
    """
    mode = str(schedule or "constant").strip().lower()
    if mode not in LR_SCHEDULE_CHOICES:
        raise ValueError(f"Unknown learning-rate schedule: {schedule!r}")
    total = max(1, int(total_steps))
    warmup = max(0, min(total - 1, int(round(total * max(0.0, min(0.25, float(warmup_ratio)))))))
    floor = max(0.01, min(1.0, float(min_lr_ratio)))

    if mode == "constant":
        return LambdaLR(optimizer, lr_lambda=lambda _step: 1.0), 0, 1.0

    warmup_start = max(floor, 0.10)

    def lr_lambda(step: int) -> float:
        current = max(0, int(step))
        if warmup:
            if current <= warmup:
                fraction = current / float(warmup)
                return warmup_start + (1.0 - warmup_start) * fraction
        decay_steps = max(1, total - warmup)
        progress = max(0.0, min(1.0, (current - warmup) / float(decay_steps)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return floor + (1.0 - floor) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda), warmup, floor


def optimizer_learning_rate(optimizer: Optimizer) -> float:
    """Return the first parameter group's current LR for logging."""
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0].get("lr", 0.0))


def checkpoint_completed_epoch(checkpoint_path: str | Path | None) -> int:
    """Return the number of completed epochs represented by a checkpoint.

    Easy GUI checkpoints are named ``checkpoint-epoch-N``.  The metadata
    fallback keeps the worker compatible with checkpoints copied or renamed
    by hand, while still defaulting safely to zero for a plain model folder.
    """
    if not checkpoint_path:
        return 0
    path = Path(checkpoint_path)
    match = re.search(r"checkpoint-epoch-(\d+)$", path.name, flags=re.IGNORECASE)
    if match:
        return max(0, int(match.group(1)))
    metadata_path = path / "training_state.json"
    if metadata_path.is_file():
        try:
            import json

            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            return max(0, int(payload.get("completed_epoch", 0)))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    return 0


def load_training_state(checkpoint_path: str | Path | None) -> dict:
    """Load lightweight progress metadata without loading model/optimizer data."""
    if not checkpoint_path:
        return {}
    path = Path(checkpoint_path) / "training_state.json"
    if not path.is_file():
        return {}
    try:
        import json

        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}


def restore_scheduler_position(scheduler: LambdaLR, optimizer: Optimizer, completed_steps: int) -> None:
    """Position a newly-created scheduler at a resumed optimizer step.

    Qwen Easy GUI checkpoints intentionally contain model/adapter weights and
    lightweight progress metadata, not Adam's very large moment tensors.  The
    optimizer therefore starts with fresh moments, but the LR schedule must
    continue at the checkpoint's global step instead of warming up again.
    """
    steps = max(0, int(completed_steps))
    if steps <= 0:
        return
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group.get("lr", 0.0))
    # LambdaLR's constructor has already applied step 0.  Setting the
    # internal cursor to N-1 and performing one normal step applies lambda(N)
    # without replaying N optimizer updates or emitting a scheduler warning.
    scheduler.last_epoch = steps - 1
    scheduler._step_count = max(1, steps)
    scheduler.step()
