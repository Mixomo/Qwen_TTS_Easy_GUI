"""Thread-safe terminal mirror used by the embedded Gradio console."""
from __future__ import annotations

import html
import io
import re
import sys
import threading
from collections import deque
from datetime import datetime

_LOCK = threading.RLock()
_LINES: deque[str] = deque(maxlen=1400)
_CURRENT = ""
_OVERWRITE = False


def _clean(value: object) -> str:
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", str(value).replace("\x00", "").replace("\ufffd", "")).rstrip()


def _commit(line: str) -> None:
    if line.strip():
        _LINES.append(_clean(line))


def _write(data: str) -> None:
    global _CURRENT, _OVERWRITE
    for char in str(data):
        if char == "\r":
            current = _clean(_CURRENT)
            if current.strip():
                if _LINES and ("%" in current or "it/s" in current):
                    _LINES[-1] = current
                else:
                    _LINES.append(current)
            _CURRENT = ""
            _OVERWRITE = True
        elif char == "\n":
            _commit(_CURRENT)
            _CURRENT = ""
            _OVERWRITE = False
        else:
            if _OVERWRITE:
                _CURRENT = ""
                _OVERWRITE = False
            _CURRENT += char


def log(message: object, level: str = "INFO") -> None:
    with _LOCK:
        prefix = f"{datetime.now():%H:%M:%S} [{level}] "
        for line in _clean(message).splitlines():
            _commit(prefix + line)


class _Mirror(io.TextIOBase):
    def __init__(self, original, level: str):
        self.original = original
        self.level = level

    def write(self, data: str) -> int:
        if data:
            with _LOCK:
                _write(data)
            self.original.write(data)
            self.original.flush()
        return len(data)

    def flush(self) -> None:
        self.original.flush()

    def isatty(self):
        return getattr(self.original, "isatty", lambda: False)()

    @property
    def encoding(self):
        return getattr(self.original, "encoding", "utf-8")


def install() -> None:
    if not getattr(sys.stdout, "_qwen_console_mirror", False):
        mirror = _Mirror(sys.stdout, "INFO")
        mirror._qwen_console_mirror = True
        sys.stdout = mirror
    if not getattr(sys.stderr, "_qwen_console_mirror", False):
        mirror = _Mirror(sys.stderr, "ERROR")
        mirror._qwen_console_mirror = True
        sys.stderr = mirror


def lines() -> list[str]:
    with _LOCK:
        result = list(_LINES)
        if _CURRENT.strip():
            result.append(_clean(_CURRENT))
        return result


def _line_color(line: str) -> str:
    lower = line.lower()
    if any(token in lower for token in ("traceback", "error", "exception", "failed", "fatal")):
        return "#f87171"
    if "warn" in lower or "fallback" in lower:
        return "#fbbf24"
    if "[train]" in lower:
        return "#c084fc"
    if any(token in lower for token in ("saved", "complete", "done", "ready")):
        return "#4ade80"
    return "#cccccc"


def html_view(title: str = "Qwen TTS Easy GUI Console") -> str:
    display = lines()[-160:] or ["Idle. Waiting for activity."]
    rows = "".join(
        f'<div style="color:{_line_color(line)};white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.5">{html.escape(line)}</div>'
        for line in display
    )
    srcdoc = (
        "<!doctype html><html><head><style>"
        "html,body{margin:0;background:#111;color:#ccc;font-family:Consolas,ui-monospace,monospace;font-size:12px;}"
        "#wrap{height:285px;border-radius:8px;border:1px solid #333;overflow:hidden;box-sizing:border-box;}"
        "#body{height:285px;overflow:auto;padding:9px 18px 9px 12px;box-sizing:border-box;scrollbar-width:thin;}"
        f"</style></head><body><div id='wrap'><div id='body' aria-label='{html.escape(title)}'>{rows}</div></div>"
        "<script>const b=document.getElementById('body'); if(!window._qwenPaused)b.scrollTop=b.scrollHeight;"
        "b.onscroll=()=>{window._qwenPaused=!(b.scrollTop+b.clientHeight>=b.scrollHeight-40)};"
        "setTimeout(()=>{if(!window._qwenPaused)b.scrollTop=b.scrollHeight},30);</script></body></html>"
    )
    return f'<iframe title="{html.escape(title)}" style="display:block;width:100%;height:285px;border:0;border-radius:8px" srcdoc="{html.escape(srcdoc, quote=True)}"></iframe>'


install()
