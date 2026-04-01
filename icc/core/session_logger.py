from __future__ import annotations

import logging
import queue
from datetime import datetime
from pathlib import Path
from threading import Thread
from time import perf_counter


logger = logging.getLogger(__name__)


class SessionLogWriter:
    def __init__(self) -> None:
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        self.path = logs_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.path.touch(exist_ok=True)
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._started_at = perf_counter()
        self._closed = False
        self._worker = Thread(target=self._writer_loop, daemon=True)
        self._worker.start()

    def append_block(self, block: str) -> None:
        if self._closed:
            return
        self._queue.put(block)

    def close(self, total_requests: int) -> None:
        if self._closed:
            return
        duration_seconds = perf_counter() - self._started_at
        self._closed = True
        self._queue.put(
            f"[SESSION END | total_requests={total_requests} | duration={duration_seconds:.1f}s]\n"
        )
        self._queue.put(None)
        self._worker.join()

    def _writer_loop(self) -> None:
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                while True:
                    block = self._queue.get()
                    if block is None:
                        self._queue.task_done()
                        break
                    handle.write(block)
                    handle.flush()
                    self._queue.task_done()
        except OSError as exc:
            logger.warning("Session log writer stopped: %s", exc)
