from __future__ import annotations

import json
import threading
from urllib.parse import urlencode

from websocket import ABNF, WebSocketApp

from icc.stt.provider import SttProvider
from icc.stt.types import EventCallback, SttConfig, TranscriptEvent, now_ms


class AssemblyAIProvider(SttProvider):
    websocket_url = "wss://streaming.assemblyai.com/v3/ws"
    VALID_MODELS = {"nano", "slam-1"}

    def __init__(self, config: SttConfig) -> None:
        self.config = config
        self.callback: EventCallback | None = None
        self.ws_app: WebSocketApp | None = None
        self.ws_thread: threading.Thread | None = None
        self.session_id: str | None = None
        self.connected_event = threading.Event()
        self._stop_lock = threading.Lock()
        self._stopped = False

    def set_event_callback(self, callback: EventCallback) -> None:
        self.callback = callback

    def connect(self) -> None:
        self._reset_session()

    def start(self) -> None:
        self._reset_session()
        if not self.config.api_key or not self.config.api_key.strip():
            raise ValueError("AssemblyAI API key is missing or empty")
        self.ws_app = WebSocketApp(
            self._build_connection_url(),
            header={"Authorization": self.config.api_key},
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self.ws_thread = threading.Thread(target=self.ws_app.run_forever, daemon=True)
        self.ws_thread.start()

    def send_audio(self, chunk: bytes) -> None:
        # Audio must only flow after the server acknowledges the stream with Begin.
        # Callers should wait on connected_event before starting their recorder loop:
        #   if provider.connected_event.wait(timeout=5):
        #       recorder.start()
        if not self.connected_event.is_set():
            return

        if self.ws_app is None:
            return

        self.ws_app.send(chunk, opcode=ABNF.OPCODE_BINARY)

    def stop(self) -> None:
        with self._stop_lock:
            # Repeated stop() calls become no-ops once shutdown has started.
            if self._stopped:
                return

            # Mark stopped before touching the socket so no late Turn/Error callbacks leak through.
            self._stopped = True
            ws_app = self.ws_app
            ws_thread = self.ws_thread
            should_send_terminate = self.connected_event.is_set() and ws_app is not None

        if ws_app is not None:
            # Early-stop fix: only send Terminate after Begin has arrived.
            if should_send_terminate:
                try:
                    ws_app.send(json.dumps({"type": "Terminate"}))
                except Exception:
                    pass
            ws_app.close()

        if ws_thread is not None:
            ws_thread.join(timeout=2)

        self._reset_local_connection()

    def close(self) -> None:
        self.stop()

    def _reset_session(self) -> None:
        self.session_id = None
        self.connected_event.clear()
        with self._stop_lock:
            self._stopped = False

    def _reset_local_connection(self) -> None:
        self.ws_app = None
        self.ws_thread = None
        self.connected_event.clear()

    def _build_connection_url(self) -> str:
        if self.config.model not in self.VALID_MODELS:
            raise ValueError(
                f"Invalid speech_model '{self.config.model}'. "
                f"Must be one of: {self.VALID_MODELS}"
            )

        query_params: list[tuple[str, str | int]] = [
            ("sample_rate", self.config.sample_rate),
            ("encoding", "pcm_s16le"),
            ("speech_model", self.config.model),
            ("format_turns", "false"),
        ]

        for keyterm in self._resolved_keyterms():
            query_params.append(("keyterms_prompt", keyterm))

        return f"{self.websocket_url}?{urlencode(query_params, doseq=True)}"

    def _resolved_keyterms(self) -> list[str]:
        if not self.config.enable_keyterms or not self.config.keyterms:
            return []
        return self.config.keyterms

    def _on_open(self, ws: WebSocketApp) -> None:
        # on_open only means the websocket is up; recorder start must still wait for Begin.
        self._emit("status_change", text="connected")

    def _on_message(self, ws: WebSocketApp, message: str) -> None:
        payload = json.loads(message)
        message_type = str(payload.get("type", ""))

        if message_type == "Begin":
            self.session_id = str(payload.get("id", "")) or self.session_id
            # Begin is the actual server-ready signal. Unblock the caller's recorder here.
            self.connected_event.set()
            self._emit("session_started", raw=payload)
            return

        if message_type == "Turn":
            transcript = str(payload.get("transcript", ""))
            if not transcript:
                return

            # format_turns=false in the connection URL, so v3 finality comes from end_of_turn only.
            is_final = bool(payload.get("end_of_turn", False))
            event_type = "final_update" if is_final else "partial_update"
            self._emit(event_type, text=transcript, is_final=is_final, raw=payload)
            return

        if message_type == "Termination":
            self._emit_closed_once(raw=payload)
            return

        if message_type == "Error":
            self._emit(
                "error",
                error_message=str(payload.get("error", "AssemblyAI streaming error")),
                raw=payload,
            )

    def _on_error(self, ws: WebSocketApp, error: object) -> None:
        if not isinstance(error, Exception):
            close_opcode = 8
            try:
                import websocket._abnf as abnf_mod

                close_opcode = abnf_mod.OPCODE_CLOSE
            except Exception:
                pass
            if hasattr(error, "opcode") and error.opcode == close_opcode:
                return
        self._emit("error", error_message=f"AssemblyAI streaming error: {error}")

    def _on_close(self, ws: WebSocketApp, status_code: int, message: str) -> None:
        self._reset_local_connection()
        self._emit_closed_once(raw={"status_code": status_code, "message": message or ""})

    def _emit_closed_once(self, raw: dict) -> None:
        with self._stop_lock:
            # Duplicate-close fix: whichever shutdown path wins marks the provider stopped.
            # After that, _emit() suppresses all stale callbacks, including the second close path.
            if self._stopped:
                return
            self._stopped = True

        if self.callback is None:
            return

        self.callback(
            TranscriptEvent(
                event_type="status_change",
                provider="assemblyai",
                text="closed",
                is_final=False,
                session_id=self.session_id,
                error_message=None,
                timestamp_ms=now_ms(),
                raw=raw,
            )
        )

    def _emit(
        self,
        event_type: str,
        text: str = "",
        is_final: bool = False,
        error_message: str | None = None,
        raw: dict | None = None,
    ) -> None:
        # Stale-event suppression: once shutdown wins, ignore every later callback.
        if self._stopped:
            return

        if self.callback is None:
            return

        self.callback(
            TranscriptEvent(
                event_type=event_type,
                provider="assemblyai",
                text=text,
                is_final=is_final,
                session_id=self.session_id,
                error_message=error_message,
                timestamp_ms=now_ms(),
                raw=raw or {},
            )
        )
