from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ROOT / "projects"
PROJECTS.mkdir(exist_ok=True)


def safe_name(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip()).strip("._-")
    if not value:
        raise ValueError("Project name is required.")
    return value


def _path(name: str) -> Path:
    return PROJECTS / safe_name(name) / "project.json"


def load_project(name: str) -> dict:
    if not name:
        return {}
    path = _path(name)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def list_projects(surface: str | None = None) -> list[str]:
    result = []
    for folder in PROJECTS.iterdir():
        if not folder.is_dir() or not (folder / "project.json").is_file():
            continue
        payload = load_project(folder.name)
        if surface and payload.get(f"{surface}_deleted"):
            continue
        if payload.get("dataset_deleted") and payload.get("training_deleted"):
            continue
        result.append(folder.name)
    return sorted(result, key=str.casefold)


def save_project(name: str, data: dict | None = None) -> str:
    name = safe_name(name)
    existing = load_project(name)
    payload = {**existing, **(data or {})}
    payload.update({"schema_version": 1, "project": name, "updated_at": datetime.now(timezone.utc).isoformat()})
    payload.setdefault("dataset", existing.get("dataset", {}))
    payload.setdefault("training", existing.get("training", {}))
    payload.setdefault("dataset_deleted", False)
    payload.setdefault("training_deleted", False)
    _write(_path(name), payload)
    return name


def update_surface(name: str, surface: str, values: dict) -> str:
    if surface not in {"dataset", "training"}:
        raise ValueError("Unknown project surface.")
    payload = load_project(name)
    payload[surface] = dict(values or {})
    payload[f"{surface}_deleted"] = False
    return save_project(name, payload)


def delete_surface(name: str, surface: str) -> str:
    payload = load_project(name)
    if not payload:
        return f"Project '{name}' does not exist."
    payload[f"{surface}_deleted"] = True
    save_project(name, payload)
    if payload.get("dataset_deleted") and payload.get("training_deleted"):
        shutil.rmtree(PROJECTS / safe_name(name), ignore_errors=True)
    return f"Removed '{name}' from {surface.title()} projects."


def clone_project(source: str, destination: str | None = None, surface: str | None = None) -> str:
    source = safe_name(source)
    if not (_path(source)).is_file():
        raise ValueError("Source project not found.")
    suffix = re.search(r"[-_ ](\d+)$", source)
    base = source[:suffix.start()] if suffix else source
    start_number = int(suffix.group(1)) + 1 if suffix else 2
    used = set(list_projects(surface)) if surface else set(list_projects())
    destination = safe_name(destination) if destination else next((f"{base}-{i}" for i in range(start_number, 10000) if f"{base}-{i}" not in used), f"{base}-copy")
    if destination in used:
        raise ValueError("Destination project already exists.")
    destination_path = PROJECTS / destination
    if destination_path.exists():
        if not surface or not (_path(destination)).is_file():
            raise ValueError("Destination project already exists.")
        # A project can remain in the other Easy GUI surface after being
        # removed from this one. Reuse the freed number while preserving that
        # other surface instead of copying over the complete project record.
        payload = load_project(destination)
        source_payload = load_project(source)
        payload[surface] = dict(source_payload.get(surface, {}))
        payload[f"{surface}_deleted"] = False
        payload.update({"project": destination, "cloned_from": source})
        _write(_path(destination), payload)
        return destination
    shutil.copytree(PROJECTS / source, PROJECTS / destination)
    payload = load_project(destination)
    payload.update({"project": destination, "cloned_from": source})
    _write(_path(destination), payload)
    return destination
