from __future__ import annotations

from abc import ABC, abstractmethod

from icc.stt.types import EventCallback


class SttProvider(ABC):
    @abstractmethod
    def set_event_callback(self, callback: EventCallback) -> None:
        pass

    @abstractmethod
    def connect(self) -> None:
        pass

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def send_audio(self, chunk: bytes) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass
