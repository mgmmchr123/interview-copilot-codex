from __future__ import annotations

from threading import Lock
from typing import Callable

from config import AppConfig
from icc.stt.service import create_stt_provider
from icc.stt.transcript_state import TranscriptState
from icc.stt.types import TranscriptEvent


ControllerCallback = Callable[[TranscriptEvent, TranscriptState], None]


class SttController:
    def __init__(self, config: AppConfig) -> None:
        self.provider = create_stt_provider(config)
        self.transcript_state = TranscriptState()
        self.callback: ControllerCallback | None = None
        self.state_lock = Lock()
        self.provider_lock = Lock()
        self.provider.set_event_callback(self._handle_provider_event)

    def set_event_callback(self, callback: ControllerCallback) -> None:
        self.callback = callback

    def connect(self) -> None:
        with self.provider_lock:
            self.provider.connect()

    def start(self) -> None:
        with self.provider_lock:
            self.provider.start()

    def send_audio(self, chunk: bytes) -> None:
        with self.provider_lock:
            self.provider.send_audio(chunk)

    def stop(self) -> None:
        with self.provider_lock:
            self.provider.stop()

    def close(self) -> None:
        with self.provider_lock:
            self.provider.close()

    def get_final_transcript(self) -> str:
        with self.state_lock:
            return self.transcript_state.final_text

    def _handle_provider_event(self, event: TranscriptEvent) -> None:
        with self.state_lock:
            state = self.transcript_state.apply_event(event)

        if self.callback is not None:
            self.callback(event, state)
