from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Any, Callable


EventCallback = Callable[["TranscriptEvent"], None]


@dataclass(slots=True)
class TranscriptEvent:
    event_type: str
    provider: str
    text: str = ""
    is_final: bool = False
    session_id: str | None = None
    error_message: str | None = None
    timestamp_ms: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SttConfig:
    provider: str
    api_key: str
    model: str
    sample_rate: int
    channels: int
    language: str
    enable_keyterms: bool = False
    keyterms: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)


def now_ms() -> int:
    return int(time() * 1000)
