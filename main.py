import argparse
import logging
import os
from threading import Thread

from config import AppConfig
from icc.deepgram_utils import check_deepgram_balance
from icc.core.orchestrator import InterviewOrchestrator
from icc.core.stt_controller import SttController
from icc.audio.recorder import AudioRecorder
from icc.llm.client import LlmClient
from icc.ui.copilot_ui import CopilotWindow
from icc.vision.camera_manager import CameraManager


logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        default="default",
        help="Resume name to load from resumes/ directory (without .json)",
    )
    args = parser.parse_args()
    config = AppConfig.from_env()
    check_deepgram_balance(
        api_key=config.deepgram_api_key,
        warning_threshold=config.deepgram_balance_warning_threshold,
    )
    camera_manager = CameraManager(config.camera_index)
    Thread(target=camera_manager.warmup, daemon=True).start()
    llm_client = LlmClient(config=config)
    orchestrator = InterviewOrchestrator(llm_client=llm_client)
    resume_path = f"resumes/{args.resume}.json"
    if os.path.exists(resume_path):
        orchestrator.load_resume(resume_path)
    else:
        logger.warning("Resume file not found: %s", resume_path)
    stt_controller = SttController(config=config)
    recorder = AudioRecorder(
        on_audio_chunk=stt_controller.send_audio,
        sample_rate=config.stt_sample_rate,
        channels=config.stt_channels,
    )

    app = CopilotWindow(
        orchestrator=orchestrator,
        stt_controller=stt_controller,
        recorder=recorder,
        config=config,
        camera_manager=camera_manager,
    )
    try:
        app.run()
    finally:
        orchestrator.shutdown()
        camera_manager.release()


if __name__ == "__main__":
    main()
