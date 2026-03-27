from __future__ import annotations

from config import AppConfig
from icc.stt.assemblyai_provider import AssemblyAIProvider
from icc.stt.deepgram_provider import DeepgramProvider
from icc.stt.provider import SttProvider
from icc.stt.types import SttConfig


def create_stt_provider(app_config: AppConfig) -> SttProvider:
    provider_name = app_config.stt_provider.lower()

    if provider_name == "assemblyai":
        if not app_config.assemblyai_api_key:
            raise ValueError("ASSEMBLYAI_API_KEY is required when STT_PROVIDER=assemblyai.")
        if app_config.assemblyai_model not in {"nano", "slam-1"}:
            raise ValueError(
                "ASSEMBLYAI_MODEL must be 'nano' or 'slam-1'."
            )
        return AssemblyAIProvider(
            SttConfig(
                provider="assemblyai",
                api_key=app_config.assemblyai_api_key,
                model=app_config.assemblyai_model,
                sample_rate=app_config.stt_sample_rate,
                channels=app_config.stt_channels,
                language=app_config.stt_language,
                enable_keyterms=app_config.assemblyai_enable_keyterms,
                keyterms=app_config.assemblyai_keyterms,
            )
        )

    if provider_name == "deepgram":
        if not app_config.deepgram_api_key:
            raise ValueError("DEEPGRAM_API_KEY is required when STT_PROVIDER=deepgram.")
        if not app_config.deepgram_model:
            raise ValueError("DEEPGRAM_MODEL is required when STT_PROVIDER=deepgram.")

        # Switching providers is config-only. The UI still receives the same
        # normalized transcript events regardless of which provider is active.
        return DeepgramProvider(
            SttConfig(
                provider="deepgram",
                api_key=app_config.deepgram_api_key,
                model=app_config.deepgram_model,
                sample_rate=app_config.stt_sample_rate,
                channels=app_config.stt_channels,
                language=app_config.stt_language,
            )
        )

    raise ValueError(f"Unsupported STT provider: {app_config.stt_provider}")
