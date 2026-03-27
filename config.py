from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
STT_PROVIDER = "deepgram"
ASSEMBLYAI_MODEL = "nano"
ASSEMBLYAI_ENABLE_KEYTERMS = True
ASSEMBLYAI_KEYTERMS: list[str] = []
DEEPGRAM_MODEL = "nova-3"
DEEPGRAM_BALANCE_WARNING_THRESHOLD = 1.0
STT_SAMPLE_RATE = 16000
STT_CHANNELS = 1
STT_LANGUAGE = "en"
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
CAMERA_CROP = float(os.getenv("CAMERA_CROP", "0.8"))
CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "1920"))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "1080"))
RESUME_CONTEXT = """
Name: [Your Name]
Current Role: [Your Role]
Years of Experience: [X]
Recent Project: [Brief description]
Tech Stack: [Your main skills]
Education: [Your education]
""".strip()


@dataclass(slots=True)
class AppConfig:
    llm_provider: str
    ollama_base_url: str
    ollama_model: str
    openai_api_key: str
    openai_model: str
    stt_provider: str
    assemblyai_api_key: str
    assemblyai_model: str
    assemblyai_enable_keyterms: bool
    assemblyai_keyterms: list[str]
    deepgram_api_key: str
    deepgram_model: str
    deepgram_balance_warning_threshold: float
    stt_sample_rate: int
    stt_channels: int
    stt_language: str
    camera_index: int
    camera_crop: float
    camera_width: int
    camera_height: int
    resume_context: str
    window_title: str = "Interview Copilot"
    window_geometry: str = "350x250"

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            llm_provider=os.getenv("LLM_PROVIDER", LLM_PROVIDER).strip() or LLM_PROVIDER,
            ollama_base_url=(
                os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL).strip()
                or OLLAMA_BASE_URL
            ),
            ollama_model=os.getenv("OLLAMA_MODEL", OLLAMA_MODEL).strip() or OLLAMA_MODEL,
            openai_api_key=os.getenv("OPENAI_API_KEY", OPENAI_API_KEY).strip(),
            openai_model=os.getenv("OPENAI_MODEL", OPENAI_MODEL).strip() or OPENAI_MODEL,
            stt_provider=os.getenv("STT_PROVIDER", STT_PROVIDER).strip() or STT_PROVIDER,
            assemblyai_api_key=os.getenv("ASSEMBLYAI_API_KEY", "").strip(),
            assemblyai_model=(
                os.getenv("ASSEMBLYAI_MODEL", ASSEMBLYAI_MODEL).strip() or ASSEMBLYAI_MODEL
            ),
            assemblyai_enable_keyterms=_parse_bool(
                os.getenv(
                    "ASSEMBLYAI_ENABLE_KEYTERMS",
                    str(ASSEMBLYAI_ENABLE_KEYTERMS),
                )
            ),
            assemblyai_keyterms=_parse_list(
                os.getenv("ASSEMBLYAI_KEYTERMS", json.dumps(ASSEMBLYAI_KEYTERMS))
            ),
            deepgram_api_key=os.getenv("DEEPGRAM_API_KEY", "").strip(),
            deepgram_model=os.getenv("DEEPGRAM_MODEL", DEEPGRAM_MODEL).strip() or DEEPGRAM_MODEL,
            deepgram_balance_warning_threshold=float(
                os.getenv(
                    "DEEPGRAM_BALANCE_WARNING_THRESHOLD",
                    str(DEEPGRAM_BALANCE_WARNING_THRESHOLD),
                )
            ),
            stt_sample_rate=int(os.getenv("STT_SAMPLE_RATE", str(STT_SAMPLE_RATE))),
            stt_channels=int(os.getenv("STT_CHANNELS", str(STT_CHANNELS))),
            stt_language=os.getenv("STT_LANGUAGE", STT_LANGUAGE).strip() or STT_LANGUAGE,
            camera_index=int(os.getenv("CAMERA_INDEX", str(CAMERA_INDEX))),
            camera_crop=float(os.getenv("CAMERA_CROP", str(CAMERA_CROP))),
            camera_width=int(os.getenv("CAMERA_WIDTH", str(CAMERA_WIDTH))),
            camera_height=int(os.getenv("CAMERA_HEIGHT", str(CAMERA_HEIGHT))),
            resume_context=RESUME_CONTEXT,
        )


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_list(value: str) -> list[str]:
    text = value.strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = [item.strip() for item in text.split(",") if item.strip()]

    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]

    return [str(parsed).strip()] if str(parsed).strip() else []
