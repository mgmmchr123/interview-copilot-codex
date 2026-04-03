from pathlib import Path

from config.interview_context import CONTEXT_PROFILES
from icc.core.orchestrator import InterviewOrchestrator
from icc.core.resume_loader import DEFAULT_INTERVIEW_CONTEXT, load_resume_profile


class _DummyLlmClient:
    config = None


def test_load_resume_profile_reads_interview_context() -> None:
    profile = load_resume_profile(str(Path("resumes") / "mta.json"))

    assert profile.filename == "mta.json"
    assert profile.company_name == "MTA"
    assert profile.interview_context == "enterprise"
    assert "Name: Jiaxin (Jerry) Qian" in profile.formatted_resume


def test_load_resume_profile_falls_back_to_default_context(tmp_path: Path) -> None:
    resume_path = tmp_path / "resume.json"
    resume_path.write_text('{"name": "Test Candidate"}', encoding="utf-8")

    profile = load_resume_profile(str(resume_path))

    assert profile.interview_context == DEFAULT_INTERVIEW_CONTEXT


def test_build_prompt_appends_answer_style_guidance() -> None:
    orchestrator = InterviewOrchestrator(llm_client=_DummyLlmClient())
    orchestrator.load_resume(str(Path("resumes") / "default.json"))

    system_message, _ = orchestrator.build_prompt(
        prompt="Tell me about a system you built.",
        answer_mode="standard",
        question_type="SYSTEM_DESIGN",
    )

    assert "## Answer Style Guidance" in system_message
    assert CONTEXT_PROFILES["growth_tech"]["style_prompt"] in system_message


def test_session_log_block_starts_with_resume_company() -> None:
    orchestrator = InterviewOrchestrator(llm_client=_DummyLlmClient())
    orchestrator.load_resume(str(Path("resumes") / "betterment.json"))

    captured: list[str] = []
    orchestrator._session_log_writer.append_block = captured.append
    orchestrator._append_session_log(
        request_id=1,
        question_type="SYSTEM_DESIGN",
        transcript="How would you design it?",
        response_text="I would start with the write path.",
    )

    assert captured
    assert captured[0].startswith("RESUME_COMPANY: Betterment\n")
