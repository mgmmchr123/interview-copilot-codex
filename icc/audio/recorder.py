from __future__ import annotations

from typing import Callable

import sounddevice as sd


AudioChunkCallback = Callable[[bytes], None]


class AudioRecorder:
    def __init__(
        self,
        on_audio_chunk: AudioChunkCallback,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024,
    ) -> None:
        self.on_audio_chunk = on_audio_chunk
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.stream: sd.InputStream | None = None

    def start(self) -> None:
        if self.stream is not None:
            return

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=self.chunk_size,
            callback=self._handle_audio,
        )
        self.stream.start()

    def stop(self) -> None:
        if self.stream is None:
            return

        self.stream.stop()
        self.stream.close()
        self.stream = None

    def _handle_audio(self, indata, frames, time, status) -> None:  # type: ignore[no-untyped-def]
        if status:
            return

        self.on_audio_chunk(indata.tobytes())
