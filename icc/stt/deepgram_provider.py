from __future__ import annotations

import json
import threading
from urllib.parse import urlencode
from uuid import uuid4

from websocket import ABNF, WebSocketApp

from icc.stt.provider import SttProvider
from icc.stt.types import EventCallback, SttConfig, TranscriptEvent, now_ms


class DeepgramProvider(SttProvider):
    websocket_url = "wss://api.deepgram.com/v1/listen"
    streaming_model = "nova-3"

    def __init__(self, config: SttConfig) -> None:
        self.config = config
        self.callback: EventCallback | None = None
        self.ws_app: WebSocketApp | None = None
        self.ws_thread: threading.Thread | None = None
        self.session_id: str | None = None
        self.connected_event = threading.Event()
        self._stop_lock = threading.Lock()
        self._stopped = False
        self._buffered_text = ""

    def set_event_callback(self, callback: EventCallback) -> None:
        self.callback = callback

    def connect(self) -> None:
        self._reset_session()

    def start(self) -> None:
        self._reset_session()
        url = self._build_connection_url()
        self.ws_app = WebSocketApp(
            url,
            header={"Authorization": f"Token {self.config.api_key}"},
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self.ws_thread = threading.Thread(target=self.ws_app.run_forever, daemon=True)
        self.ws_thread.start()

    def send_audio(self, chunk: bytes) -> None:
        if not self.connected_event.is_set():
            return

        if self.ws_app is None:
            return

        self.ws_app.send(chunk, opcode=ABNF.OPCODE_BINARY)

    def stop(self) -> None:
        with self._stop_lock:
            if self._stopped:
                return

            self._stopped = True
            ws_app = self.ws_app
            ws_thread = self.ws_thread
            should_send_close = self.connected_event.is_set() and ws_app is not None

        if ws_app is not None:
            if should_send_close:
                try:
                    ws_app.send(json.dumps({"type": "CloseStream"}))
                except Exception:
                    pass
            ws_app.close()

        if ws_thread is not None:
            ws_thread.join(timeout=2)

        self._reset_local_connection()

    def close(self) -> None:
        self.stop()

    def _reset_session(self) -> None:
        self.session_id = str(uuid4())
        self.connected_event.clear()
        self._buffered_text = ""
        with self._stop_lock:
            self._stopped = False

    def _reset_local_connection(self) -> None:
        self.ws_app = None
        self.ws_thread = None
        self.connected_event.clear()

    def _build_connection_url(self) -> str:
        query_params: list[tuple[str, str | int]] = [
            ("encoding", "linear16"),
            ("sample_rate", self.config.sample_rate),
            ("model", self.streaming_model),
            ("language", "en"),
            ("interim_results", "true"),
            ("utterance_end_ms", "1000"),
            ("vad_events", "true"),
        ]
        return f"{self.websocket_url}?{urlencode(query_params, doseq=True)}"

    def _on_open(self, ws: WebSocketApp) -> None:
        self.connected_event.set()
        self._emit("status_change", text="connected")
        self._emit("session_started")

    def _on_message(self, ws: WebSocketApp, message: str) -> None:
        payload = json.loads(message)
        message_type = str(payload.get("type", ""))

        if message_type == "Results":
            transcript = self._extract_transcript(payload)
            if not transcript:
                return

            self._buffered_text = transcript

            if bool(payload.get("speech_final", False)):
                self._emit("final_update", text=transcript, is_final=True, raw=payload)
                self._buffered_text = ""
                return

            if not bool(payload.get("is_final", False)):
                self._emit("partial_update", text=transcript, is_final=False, raw=payload)
            return

        if message_type == "UtteranceEnd":
            if self._buffered_text:
                self._emit(
                    "final_update",
                    text=self._buffered_text,
                    is_final=True,
                    raw=payload,
                )
                self._buffered_text = ""
            return

    def _on_error(self, ws: WebSocketApp, error: object) -> None:
        self._emit("error", error_message=f"Deepgram streaming error: {error}")

    def _on_close(self, ws: WebSocketApp, status_code: int, message: str) -> None:
        self._reset_local_connection()
        self._emit_closed_once(raw={"status_code": status_code, "message": message or ""})

    def _emit_closed_once(self, raw: dict) -> None:
        with self._stop_lock:
            if self._stopped:
                return
            self._stopped = True

        if self.callback is None:
            return

        self.callback(
            TranscriptEvent(
                event_type="status_change",
                provider="deepgram",
                text="closed",
                is_final=False,
                session_id=self.session_id,
                error_message=None,
                timestamp_ms=now_ms(),
                raw=raw,
            )
        )

    def _extract_transcript(self, payload: dict) -> str:
        channel = payload.get("channel", {})
        alternatives = channel.get("alternatives", [])
        if not alternatives:
            return ""
        return str(alternatives[0].get("transcript", "")).strip()

    def _emit(
        self,
        event_type: str,
        text: str = "",
        is_final: bool = False,
        error_message: str | None = None,
        raw: dict | None = None,
    ) -> None:
        if self._stopped:
            return

        if self.callback is None:
            return

        self.callback(
            TranscriptEvent(
                event_type=event_type,
                provider="deepgram",
                text=text,
                is_final=is_final,
                session_id=self.session_id,
                error_message=error_message,
                timestamp_ms=now_ms(),
                raw=raw or {},
            )
        )
