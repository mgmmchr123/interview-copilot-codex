from __future__ import annotations

import asyncio
import logging
import os
import queue
import tkinter as tk
from datetime import datetime
from threading import Thread
from time import perf_counter
from tkinter import messagebox, scrolledtext
from typing import Callable

import cv2
from PIL import Image, ImageTk

from config import AppConfig
from icc.core.orchestrator import InterviewOrchestrator
from icc.core.stt_controller import SttController
from icc.audio.recorder import AudioRecorder
from icc.stt.transcript_state import TranscriptState
from icc.stt.types import TranscriptEvent
from icc.vision.camera_manager import CameraManager
from icc.vision.screenshot import image_to_base64, save_capture_image


logger = logging.getLogger(__name__)
UiMessage = tuple[str, int, str]
QUEUE_POLL_MS = 15
UI_FONT_FAMILY = "Segoe UI"
UI_FONT_SIZE = 11
ASSEMBLYAI_BEGIN_TIMEOUT_SECONDS = 10
CAMERA_PREVIEW_INTERVAL_MS = 30
THUMBNAIL_SIZE = (80, 80)
MAX_IMAGES = 3
MAX_PREVIEW_SIZE = (640, 480)
CAMERA_DIALOG_X_OFFSET = 120
CAMERA_DIALOG_BUTTON_TOP_PAD = 80
DEFAULT_CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "1920"))
DEFAULT_CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "1080"))
ANSWER_AREA_EXTRA_HEIGHT_PX = 40


class CameraCaptureDialog:
    def __init__(
        self,
        parent: tk.Tk,
        camera_manager: CameraManager,
        crop_ratio: float,
        on_capture: Callable[[Image.Image], None],
    ) -> None:
        self.parent = parent
        self.camera_manager = camera_manager
        self.crop_ratio = max(0.1, min(crop_ratio, 1.0))
        self.on_capture = on_capture
        self.window = tk.Toplevel(parent)
        self.window.title("Camera Preview")
        self.window.transient(parent)
        self.window.resizable(False, False)
        self.window.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self._preview_size: tuple[int, int] | None = None

        self.right_panel = tk.Frame(self.window, padx=12, pady=12)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        self.preview_label = tk.Label(self.window, bg="black")
        self.preview_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(12, 0), pady=12)
        self.capture_button = tk.Button(
            self.right_panel,
            text="\U0001f4f8 \u62cd\u6444",
            command=self._on_capture,
            width=10,
            font=(UI_FONT_FAMILY, 10),
        )
        self.cancel_button = tk.Button(
            self.right_panel,
            text="\u2715 \u53d6\u6d88",
            command=self._on_cancel,
            width=10,
            font=(UI_FONT_FAMILY, 10),
        )
        self.capture_button.pack(side=tk.TOP, pady=(CAMERA_DIALOG_BUTTON_TOP_PAD, 8))
        self.cancel_button.pack(side=tk.TOP)

        self.cap = self.camera_manager.get_cap()
        self._preview_job: str | None = None
        self._latest_frame_rgb: Image.Image | None = None
        self._preview_photo: ImageTk.PhotoImage | None = None
        self._is_centered = False

        if self.cap is None or not self.cap.isOpened():
            self._cleanup()
            messagebox.showerror(
                "Camera Error",
                f"\u274c Could not open camera at index {self.camera_manager.camera_index}. "
                "Try setting CAMERA_INDEX=1 or CAMERA_INDEX=2",
                parent=parent,
            )
            self.window.destroy()
            return

        self.cap.read()
        self._preview_size = self._compute_preview_size()
        self.preview_label.config(anchor="center")
        self.window.grab_set()
        self.window.focus_set()
        self._update_preview()

    def _update_preview(self) -> None:
        if not self.window.winfo_exists():
            return

        ret, frame = self.cap.read()
        if ret and frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._latest_frame_rgb = Image.fromarray(frame_rgb)
            preview_image = self._latest_frame_rgb
            if self._preview_size is not None and preview_image.size != self._preview_size:
                preview_image = preview_image.resize(self._preview_size, Image.LANCZOS)
            self._preview_photo = ImageTk.PhotoImage(preview_image)
            self.preview_label.config(image=self._preview_photo)
            if not self._is_centered:
                self._center_window()
                self._is_centered = True

        self._preview_job = self.window.after(CAMERA_PREVIEW_INTERVAL_MS, self._update_preview)

    def _center_window(self) -> None:
        self.window.update_idletasks()
        w = self.window.winfo_width()
        h = self.window.winfo_height()
        sw = self.window.winfo_screenwidth()
        sh = self.window.winfo_screenheight()
        x = max(0, (sw - w) // 2 - CAMERA_DIALOG_X_OFFSET)
        y = max(0, (sh - h) // 2)
        self.window.geometry(f"+{x}+{y}")

    def _on_capture(self) -> None:
        if self._latest_frame_rgb is None:
            messagebox.showerror(
                "Camera Error",
                "Failed to capture frame from camera.",
                parent=self.window,
            )
            return

        captured_image = self._latest_frame_rgb.copy()
        self._cleanup()
        self.window.destroy()
        self.on_capture(captured_image)

    def _on_cancel(self) -> None:
        self._cleanup()
        self.window.destroy()

    def _cleanup(self) -> None:
        if self._preview_job is not None and self.window.winfo_exists():
            self.window.after_cancel(self._preview_job)
            self._preview_job = None

    def _compute_preview_size(self) -> tuple[int, int]:
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0

        if frame_width <= 0 or frame_height <= 0:
            return MAX_PREVIEW_SIZE

        scale = min(
            1.0,
            MAX_PREVIEW_SIZE[0] / frame_width,
            MAX_PREVIEW_SIZE[1] / frame_height,
        )
        return (
            max(1, int(frame_width * scale)),
            max(1, int(frame_height * scale)),
        )

    def _crop_frame(self, frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
        if self.crop_ratio >= 1.0:
            return frame

        height, width = frame.shape[:2]
        crop_width = max(1, int(width * self.crop_ratio))
        crop_height = max(1, int(height * self.crop_ratio))
        x1 = (width - crop_width) // 2
        y1 = (height - crop_height) // 2
        return frame[y1:y1 + crop_height, x1:x1 + crop_width]


class CopilotWindow:
    def __init__(
        self,
        orchestrator: InterviewOrchestrator,
        stt_controller: SttController,
        recorder: AudioRecorder,
        config: AppConfig,
        camera_manager: CameraManager,
    ) -> None:
        self.orchestrator = orchestrator
        self.stt_controller = stt_controller
        self.recorder = recorder
        self.config = config
        self.camera_manager = camera_manager
        self.message_queue: queue.Queue[UiMessage] = queue.Queue()
        self.debug_stream = os.getenv("ICC_DEBUG_STREAM", "").strip() == "1"
        self._request_started_at: float | None = None
        self._next_request_id = 0
        self._active_request_id: int | None = None
        self._asyncio_loop = asyncio.new_event_loop()
        self._asyncio_thread = Thread(target=self._run_asyncio_loop, daemon=True)
        self._asyncio_thread.start()
        self._stt_state = "idle"
        self._llm_state = "idle"
        self._latest_transcript_text = ""
        self.in_coding_session = False
        self._pre_request_output_text = ""
        self._pre_request_was_coding_session = False
        self._first_token_received: dict[int, bool] = {}
        self._full_response_by_request: dict[int, str] = {}
        self._thinking_timeout_tasks: dict[int, asyncio.Task[None]] = {}
        self.captured_images_b64: list[str] = []
        self.captured_image_paths: list[str] = []
        self._captured_thumbnail_photos: list[ImageTk.PhotoImage] = []
        self._rendered_thumbnail_count = 0
        self._camera_dialog: CameraCaptureDialog | None = None

        self.root = tk.Tk()
        self.root.title(config.window_title)
        width, height = self._parse_window_size(config.window_geometry)
        self._center_window(width, height + ANSWER_AREA_EXTRA_HEIGHT_PX)
        self.root.attributes("-topmost", True)
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.container = tk.Frame(self.root, padx=12, pady=12)
        self.prompt_frame = tk.Frame(self.container)
        self.output_frame = tk.Frame(self.container)
        self.footer_frame = tk.Frame(self.container)
        self.button_row_top = tk.Frame(self.footer_frame)
        self.button_row_bottom = tk.Frame(self.footer_frame)
        self.mode_frame = tk.Frame(self.button_row_bottom)
        self._has_streamed_content = False
        self._user_at_bottom = True
        self._last_chunk_request_id: int | None = None
        self._last_chunk_payload = ""
        self.answer_mode_var = tk.StringVar(value="Grounded Senior (FINRA-grounded, defend-ready)")
        self.prompt_input = scrolledtext.ScrolledText(
            self.prompt_frame,
            wrap=tk.WORD,
            font=(UI_FONT_FAMILY, UI_FONT_SIZE),
            height=2,
        )
        self.output = scrolledtext.ScrolledText(
            self.output_frame,
            wrap=tk.WORD,
            state=tk.DISABLED,
            font=(UI_FONT_FAMILY, UI_FONT_SIZE),
            height=6,
        )
        self._output_scrollbar_set = self.output.vbar.set
        self.output.configure(yscrollcommand=self._on_output_scroll)
        self.stop_answer_button = tk.Button(
            self.button_row_top,
            text="Answer",
            command=self._on_answer_click,
            width=8,
            font=(UI_FONT_FAMILY, 9),
            padx=6,
            pady=3,
            state=tk.DISABLED,
        )
        self.listen_button = tk.Button(
            self.button_row_top,
            text="Listen",
            command=self._on_listen_click,
            width=8,
            font=(UI_FONT_FAMILY, 9),
            padx=6,
            pady=3,
        )
        self.mode_label = tk.Label(
            self.mode_frame,
            text="Mode",
            font=(UI_FONT_FAMILY, 9),
        )
        self.answer_mode = tk.OptionMenu(
            self.mode_frame,
            self.answer_mode_var,
            "Standard",
            "Senior (Trade-offs + Failure modes)",
            "Grounded Senior (FINRA-grounded, defend-ready)",
        )
        self.answer_mode.config(
            width=8,
            font=(UI_FONT_FAMILY, 9),
        )
        self.follow_up_button = tk.Button(
            self.button_row_top,
            text="Follow-up",
            command=self._on_follow_up_click,
            width=8,
            font=(UI_FONT_FAMILY, 9),
            padx=6,
            pady=3,
            state=tk.DISABLED,
        )
        self.screenshot_button = tk.Button(
            self.button_row_bottom,
            text="Camera",
            command=self._on_screenshot_click,
            width=8,
            font=(UI_FONT_FAMILY, 9),
            padx=6,
            pady=3,
        )
        self.new_topic_button = tk.Button(
            self.button_row_bottom,
            text="Clear",
            command=self._on_new_topic_click,
            width=8,
            font=(UI_FONT_FAMILY, 9),
            padx=6,
            pady=3,
        )

        self.stt_controller.set_event_callback(self._on_stt_event)
        self._build_layout()
        self.root.after(QUEUE_POLL_MS, self._drain_queue)

    def run(self) -> None:
        self.root.mainloop()

    def _run_asyncio_loop(self) -> None:
        asyncio.set_event_loop(self._asyncio_loop)
        self._asyncio_loop.run_forever()

    def _on_close(self) -> None:
        self._cancel_all_thinking_timeouts()
        if self._camera_dialog is not None and self._camera_dialog.window.winfo_exists():
            self._camera_dialog._cleanup()
            self._camera_dialog.window.destroy()
            self._camera_dialog = None
        self._asyncio_loop.call_soon_threadsafe(self._asyncio_loop.stop)
        self.camera_manager.release()
        self.root.destroy()

    def _build_layout(self) -> None:
        self.container.pack(fill=tk.BOTH, expand=True)
        self.prompt_frame.pack(side=tk.TOP, fill=tk.X)
        self.footer_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(6, 0))
        self.output_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(6, 0))

        self.prompt_input.pack(fill=tk.X)
        self.output.pack(fill=tk.BOTH, expand=True)
        self.button_row_top.pack(anchor="center")
        self.button_row_bottom.pack(anchor="center", pady=(6, 0))
        self.listen_button.pack(side=tk.LEFT, padx=4)
        self.stop_answer_button.pack(side=tk.LEFT, padx=4)
        self.follow_up_button.pack(side=tk.LEFT, padx=4)
        self.screenshot_button.pack(side=tk.LEFT, padx=4)
        self.new_topic_button.pack(side=tk.LEFT, padx=4)
        self.mode_frame.pack(side=tk.LEFT, padx=4)
        self.mode_label.pack(side=tk.LEFT, padx=(0, 4))
        self.answer_mode.pack(side=tk.LEFT)
        self.prompt_input.insert("1.0", self.orchestrator.default_prompt)
        self._replace_output("Ready")
        self.output.see("1.0")
        self._update_controls()

    def _on_listen_click(self) -> None:
        if self._stt_state != "idle":
            return
        self._start_listening()

    def _start_listening(self) -> None:
        try:
            self.stt_controller.connect()
            self.stt_controller.start()
        except Exception as exc:
            self._replace_output(f"STT start failed:\n\n{exc}")
            self.output.see("1.0")
            return

        self._stt_state = "starting"
        self._latest_transcript_text = ""
        self._update_controls()
        self._replace_output("Connecting...")
        self.output.see("1.0")
        Thread(target=self._watch_stt_start_timeout, daemon=True).start()

    def _stop_and_ask(self) -> None:
        prompt = self._current_prompt_text()
        images_b64 = list(self.captured_images_b64)

        if self._stt_state == "listening":
            self.recorder.stop()
            self._stt_state = "idle"
            self._update_controls()
            Thread(target=self._stop_stt_session, daemon=True).start()

        if not prompt and not images_b64:
            self._replace_output("No question sent.")
            self.output.see("1.0")
            return

        self._clear_all_captured_images()
        self._clear_transcript()
        logger.info(
            "Transcript-to-LLM handoff timestamp: %s (chars=%d)",
            datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
            len(prompt),
        )
        self._start_llm_request(prompt, images_b64=images_b64)

    def _on_screenshot_click(self) -> None:
        if len(self.captured_images_b64) >= MAX_IMAGES:
            messagebox.showwarning(
                "Maximum Images",
                "Maximum 3 images allowed. Remove one first.",
                parent=self.root,
            )
            return
        if self._camera_dialog is not None and self._camera_dialog.window.winfo_exists():
            return

        self._camera_dialog = CameraCaptureDialog(
            parent=self.root,
            camera_manager=self.camera_manager,
            crop_ratio=self.config.camera_crop,
            on_capture=self._on_camera_capture_complete,
        )

    def _on_new_topic_click(self) -> None:
        self._clear_transcript()
        self._clear_all_captured_images()
        self.in_coding_session = False
        self._pre_request_output_text = ""
        self._pre_request_was_coding_session = False
        self.orchestrator.clear_history()
        self._replace_output("Ready")
        self.output.see("1.0")
        self._update_controls()

    def _stop_stt_session(self) -> None:
        try:
            self.stt_controller.stop()
        except Exception as exc:
            self._enqueue("stt_error", 0, str(exc))

    def _on_answer_click(self) -> None:
        if self._llm_state == "generating":
            return
        self._stop_and_ask()

    def _on_follow_up_click(self) -> None:
        if self._llm_state == "generating":
            return
        prompt = self._current_prompt_text()
        images_b64 = list(self.captured_images_b64)

        if self._stt_state == "listening":
            self.recorder.stop()
            self._stt_state = "idle"
            self._update_controls()
            Thread(target=self._stop_stt_session, daemon=True).start()

        if not prompt and not images_b64:
            self._replace_output("No question sent.")
            self.output.see("1.0")
            return

        self._clear_all_captured_images()
        self._clear_transcript()
        logger.info(
            "Transcript-to-LLM handoff timestamp: %s (chars=%d)",
            datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
            len(prompt),
        )
        prompt = (
            prompt
            + "\n\n[This is a follow-up to the previous answer. "
            "Treat this as FOLLOW_UP type.]"
        )
        self._start_llm_request(prompt, images_b64=images_b64)

    def _start_llm_request(
        self,
        prompt: str,
        images_b64: list[str] | None = None,
    ) -> None:
        request_id = self._start_new_request()
        mode = self.orchestrator.default_mode
        attached_images_b64 = images_b64 if images_b64 is not None else list(self.captured_images_b64)

        self.orchestrator.request_answer(
            prompt=prompt,
            mode=mode,
            images_b64=attached_images_b64,
            answer_mode=(
                "grounded_senior"
                if "Grounded Senior" in self.answer_mode_var.get()
                else "senior"
                if "Senior" in self.answer_mode_var.get()
                else "standard"
            ),
            on_chunk=lambda text, rid=request_id: self._enqueue("chunk", rid, text),
            on_complete=lambda text, rid=request_id: self._enqueue("complete", rid, text),
            on_error=lambda message, rid=request_id: self._enqueue("error", rid, message),
        )

    def _start_new_request(self) -> int:
        self._next_request_id += 1
        request_id = self._next_request_id
        self._active_request_id = request_id
        self._llm_state = "generating"
        self._request_started_at = perf_counter()
        self._has_streamed_content = False
        self._user_at_bottom = False
        self._last_chunk_request_id = request_id
        self._last_chunk_payload = ""
        self._first_token_received[request_id] = False
        self._full_response_by_request[request_id] = ""
        self._pre_request_output_text = self._current_output_text()
        self._pre_request_was_coding_session = self.in_coding_session
        self._schedule_thinking_timeout(request_id)
        logger.info(
            "LLM request start timestamp: %s (request_id=%s)",
            datetime.now().isoformat(timespec="milliseconds"),
            request_id,
        )
        self._update_controls()
        self._replace_output("Thinking...")
        self.output.yview_moveto(0.0)
        return request_id

    def _parse_window_size(self, geometry: str) -> tuple[int, int]:
        size = geometry.split("+", maxsplit=1)[0]
        width_text, height_text = size.split("x", maxsplit=1)
        return int(width_text), int(height_text)

    def _center_window(self, width: int, height: int) -> None:
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def _on_stt_event(self, event: TranscriptEvent, state: TranscriptState) -> None:
        if event.event_type == "session_started":
            self._enqueue("stt_ready", 0, "")
        elif event.event_type in ("partial_update", "final_update"):
            self._enqueue("stt_transcript", 0, state.live_text)
        elif event.event_type == "error":
            self._enqueue("stt_error", 0, state.error_message)

    def _enqueue(self, kind: str, request_id: int, payload: str) -> None:
        self._debug_log("queue_put", f"kind={kind} request_id={request_id} chars={len(payload)}")
        self.message_queue.put((kind, request_id, payload))

    def show_full_response(self, text: str) -> None:
        if not self._has_streamed_content:
            self._prepare_output_for_new_answer()
        if text:
            self._append_output(text)
            self._has_streamed_content = True

    def _schedule_thinking_timeout(self, request_id: int) -> None:
        def schedule() -> None:
            self._thinking_timeout_tasks[request_id] = asyncio.create_task(
                self._thinking_timeout(request_id)
            )

        self._asyncio_loop.call_soon_threadsafe(schedule)

    async def _thinking_timeout(self, request_id: int) -> None:
        try:
            await asyncio.sleep(3)
            self._enqueue("thinking_timeout", request_id, "")
        except asyncio.CancelledError:
            return

    def _cancel_thinking_timeout(self, request_id: int) -> None:
        task = self._thinking_timeout_tasks.pop(request_id, None)
        if task is None:
            return
        self._asyncio_loop.call_soon_threadsafe(task.cancel)

    def _cancel_all_thinking_timeouts(self) -> None:
        for request_id in list(self._thinking_timeout_tasks):
            self._cancel_thinking_timeout(request_id)

    def _cleanup_request_state(self, request_id: int) -> None:
        self._cancel_thinking_timeout(request_id)
        self._first_token_received.pop(request_id, None)
        self._full_response_by_request.pop(request_id, None)

    def _handle_silent_failure(self, request_id: int, full_text: str) -> None:
        warning = (
            "WARNING: streaming silent failure detected, "
            f"falling back to full response (request_id={request_id})"
        )
        logger.warning(warning)
        self.orchestrator.log_session_warning(warning)
        self.show_full_response(full_text)

    def _finish_request_ui(self, request_id: int) -> None:
        if request_id == self._active_request_id:
            self._active_request_id = None
            self._llm_state = "idle"
            self._last_chunk_request_id = None
            self._last_chunk_payload = ""
            self._pre_request_output_text = ""
            self._pre_request_was_coding_session = False
            self._clear_transcript()
            self._update_controls()
        self._cleanup_request_state(request_id)

    def _watch_stt_start_timeout(self) -> None:
        provider = getattr(self.stt_controller, "provider", None)
        connected_event = getattr(provider, "connected_event", None)
        if connected_event is None:
            return

        if connected_event.wait(timeout=ASSEMBLYAI_BEGIN_TIMEOUT_SECONDS):
            return

        if self._stt_state != "starting":
            return

        Thread(target=self._stop_stt_session, daemon=True).start()
        self._enqueue(
            "stt_error",
            0,
            f"AssemblyAI Begin message not received within {ASSEMBLYAI_BEGIN_TIMEOUT_SECONDS}s",
        )

    def _drain_queue(self) -> None:
        while True:
            try:
                kind, request_id, payload = self.message_queue.get_nowait()
            except queue.Empty:
                break

            if kind == "stt_ready":
                if self._stt_state == "starting":
                    try:
                        self.recorder.start()
                    except Exception as exc:
                        self._stt_state = "idle"
                        self._update_controls()
                        self._replace_output(f"Mic start failed:\n\n{exc}")
                        self.output.see("1.0")
                        continue

                    self._stt_state = "listening"
                    self._update_controls()
                    self._replace_output("Listening...")
                    self.output.see("1.0")
                continue

            if kind == "stt_transcript":
                if self._stt_state == "listening":
                    self._latest_transcript_text = payload
                    yview_before = self.prompt_input.yview()
                    self.prompt_input.delete("1.0", tk.END)
                    self.prompt_input.insert("1.0", payload)
                    if yview_before[1] >= 0.99:
                        self.prompt_input.see(tk.END)
                    else:
                        self.prompt_input.yview_moveto(yview_before[0])
                continue

            if kind == "stt_error":
                self._stt_state = "idle"
                self._update_controls()
                self._replace_output(f"STT failed:\n\n{payload}")
                self.output.see("1.0")
                continue

            if kind == "complete":
                self._debug_log("render_complete", f"request_id={request_id} done")
                full_text = payload
                if request_id == self._active_request_id:
                    if not self._first_token_received.get(request_id, False):
                        self._handle_silent_failure(request_id, full_text)
                    self._finish_request_ui(request_id)
                else:
                    self._cleanup_request_state(request_id)
                continue

            if kind == "thinking_timeout":
                if (
                    request_id == self._active_request_id
                    and not self._first_token_received.get(request_id, False)
                ):
                    self._handle_silent_failure(
                        request_id,
                        self._full_response_by_request.get(request_id, ""),
                    )
                    self._finish_request_ui(request_id)
                continue

            if request_id != self._active_request_id:
                self._debug_log(
                    "drop_stale",
                    f"kind={kind} request_id={request_id} active_request_id={self._active_request_id}",
                )
                continue

            if kind == "chunk":
                self._debug_log("render_chunk", f"request_id={request_id} chars={len(payload)}")
                if (
                    self._last_chunk_request_id == request_id
                    and payload
                    and payload == self._last_chunk_payload
                ):
                    self._debug_log(
                        "drop_duplicate_chunk",
                        f"request_id={request_id} chars={len(payload)}",
                    )
                    continue
                if not self._has_streamed_content:
                    logger.info(
                        "LLM first-token UI timestamp: %s (request_id=%s)",
                        datetime.now().isoformat(timespec="milliseconds"),
                        request_id,
                    )
                    self._cancel_thinking_timeout(request_id)
                    self._prepare_output_for_new_answer()
                self._full_response_by_request[request_id] = (
                    self._full_response_by_request.get(request_id, "") + payload
                )
                self._append_output(payload)
                self._has_streamed_content = True
                self._first_token_received[request_id] = True
                self._last_chunk_request_id = request_id
                self._last_chunk_payload = payload
            elif kind == "error":
                self._debug_log("render_error", f"request_id={request_id} chars={len(payload)}")
                self._finish_request_ui(request_id)
                self._replace_output(payload)
                self.output.see("1.0")

        self.root.after(QUEUE_POLL_MS, self._drain_queue)

    def _replace_output(self, text: str) -> None:
        self.output.config(state=tk.NORMAL)
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, text)
        self.output.config(state=tk.DISABLED)
        self._rendered_thumbnail_count = 0
        if self._should_embed_thumbnails(text):
            self._render_all_thumbnails()

    def _append_output(self, text: str) -> None:
        self.output.config(state=tk.NORMAL)
        self.output.insert(tk.END, text)
        self.output.config(state=tk.DISABLED)

    def _current_output_text(self) -> str:
        return self.output.get("1.0", tk.END).strip()

    def _prepare_output_for_new_answer(self) -> None:
        question_type = self.orchestrator.current_question_type
        should_append = question_type == "CODING" or (
            question_type == "FOLLOW_UP" and self._pre_request_was_coding_session
        )

        if should_append:
            self.in_coding_session = True
            existing_text = self._pre_request_output_text
            if existing_text in {
                "Ready",
                "Thinking...",
                "Connecting...",
                "Listening...",
                "No question sent.",
            }:
                existing_text = ""
            separator = f"--- {datetime.now().strftime('%H:%M:%S')} ---"
            new_text = f"{existing_text}\n\n{separator}\n" if existing_text else f"{separator}\n"
            self._replace_output(new_text)
            separator_line = max(1, int(self.output.index("end-1c").split(".")[0]) - 1)
            self.output.see(f"{separator_line}.0")
            return

        self.in_coding_session = False
        self._replace_output("")

    def _on_output_scroll(self, first: str, last: str) -> None:
        self._output_scrollbar_set(first, last)
        try:
            self._user_at_bottom = float(last) >= 0.99
        except ValueError:
            self._user_at_bottom = True

    def _update_controls(self) -> None:
        if self._llm_state == "generating":
            self.listen_button.config(state=tk.NORMAL)
            self.stop_answer_button.config(state=tk.DISABLED)
            self.follow_up_button.config(state=tk.DISABLED)
        elif self._stt_state == "starting":
            self.listen_button.config(state=tk.DISABLED)
            self.stop_answer_button.config(state=tk.DISABLED)
            self.follow_up_button.config(state=tk.DISABLED)
        else:
            self.listen_button.config(
                state=tk.DISABLED if self._stt_state == "listening" else tk.NORMAL
            )
            self.stop_answer_button.config(state=tk.NORMAL)
            self.follow_up_button.config(state=tk.NORMAL)

        self.screenshot_button.config(
            state=tk.DISABLED if len(self.captured_images_b64) >= MAX_IMAGES else tk.NORMAL
        )
        self.new_topic_button.config(state=tk.NORMAL)

    def _debug_log(self, stage: str, detail: str) -> None:
        if not self.debug_stream:
            return

        if self._request_started_at is None:
            elapsed_ms = 0.0
        else:
            elapsed_ms = (perf_counter() - self._request_started_at) * 1000

        print(f"[stream-debug][ui][{elapsed_ms:8.1f} ms] {stage}: {detail}")

    def _on_camera_capture_complete(self, image: Image.Image) -> None:
        self._camera_dialog = None
        capture_path = save_capture_image(image)
        thumbnail = image.copy()
        thumbnail.thumbnail(THUMBNAIL_SIZE)
        self.captured_image_paths.append(capture_path.as_posix())
        self.captured_images_b64.append(image_to_base64(image))
        self._captured_thumbnail_photos.append(ImageTk.PhotoImage(thumbnail))
        self._render_all_thumbnails()
        self._update_controls()

    def _render_all_thumbnails(self) -> None:
        self.output.config(state=tk.NORMAL)
        if self._rendered_thumbnail_count > 0:
            self.output.delete("1.0", f"{self._rendered_thumbnail_count + 1}.0")
        self._rendered_thumbnail_count = 0
        for index in reversed(range(len(self._captured_thumbnail_photos))):
            tag_name = f"remove_tag_{index}"
            self.output.tag_delete(tag_name)
            self.output.tag_config(tag_name, foreground="blue", underline=True)
            self.output.tag_bind(
                tag_name,
                "<Button-1>",
                lambda event, idx=index: self._on_remove_thumbnail_click(event, idx),
            )
            self.output.image_create("1.0", image=self._captured_thumbnail_photos[index])
            self.output.insert("1.1", "  \u2715\n", tag_name)
            self._rendered_thumbnail_count += 1
        self.output.config(state=tk.DISABLED)

    def _remove_captured_image(self, index: int) -> None:
        if index < 0 or index >= len(self.captured_images_b64):
            return
        del self.captured_images_b64[index]
        del self.captured_image_paths[index]
        del self._captured_thumbnail_photos[index]
        self._render_all_thumbnails()
        self._update_controls()

    def _clear_all_captured_images(self) -> None:
        self.captured_images_b64.clear()
        self.captured_image_paths.clear()
        self._captured_thumbnail_photos.clear()
        self._render_all_thumbnails()
        self._update_controls()

    def _clear_transcript(self) -> None:
        self._latest_transcript_text = ""
        self.prompt_input.delete("1.0", tk.END)

    def _current_prompt_text(self) -> str:
        return self.prompt_input.get("1.0", tk.END).strip()

    def _on_remove_thumbnail_click(self, event: tk.Event, index: int) -> str:
        self._remove_captured_image(index)
        return "break"

    def _should_embed_thumbnails(self, text: str) -> bool:
        if not self.captured_images_b64:
            return False
        return text not in {
            "Ready",
            "Thinking...",
            "Connecting...",
            "Listening...",
            "No question sent.",
        }
