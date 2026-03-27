from __future__ import annotations

from dataclasses import dataclass, field, replace

from icc.stt.types import TranscriptEvent, now_ms


@dataclass(slots=True)
class TranscriptState:
    live_text: str = ""
    final_text: str = ""
    status_text: str = ""
    error_message: str = ""
    last_updated_at: int | None = None
    is_final: bool = False
    active_provider: str = ""
    last_event_type: str = ""
    _last_event_id: str = field(default="", repr=False, compare=False)

    @property
    def display_text(self) -> str:
        return self.live_text

    def _apply_transcript(self, text: str, is_final: bool) -> None:
        if is_final:
            self.final_text = (self.final_text + " " + text).strip()
            self.live_text = self.final_text
        else:
            self.live_text = (self.final_text + " " + text).strip()
        self.is_final = is_final
        self.error_message = ""

    def apply_event(self, event: TranscriptEvent) -> "TranscriptState":
        event_id = f"{event.event_type}:{event.timestamp_ms}:{event.text}"
        if event_id == self._last_event_id:
            return replace(self)
        self._last_event_id = event_id

        self.active_provider = event.provider
        self.last_event_type = event.event_type
        self.last_updated_at = event.timestamp_ms or now_ms()

        if event.event_type == "session_started":
            self.live_text = ""
            self.final_text = ""
            self.error_message = ""
            self.is_final = False
        elif event.event_type == "status_change":
            self.status_text = event.text
        elif event.event_type in ("partial_update", "final_update"):
            if not event.text:
                return replace(self)
            self._apply_transcript(event.text, event.is_final)
        elif event.event_type == "error":
            self.error_message = event.error_message or event.text

        return replace(self)
