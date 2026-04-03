from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_INTERVIEW_CONTEXT = "growth_tech"


@dataclass(frozen=True, slots=True)
class ResumeProfile:
    filename: str
    company_name: str
    formatted_resume: str
    interview_context: str


def load_resume(path: str) -> str:
    """Load a resume JSON file and return formatted plain text."""
    return load_resume_profile(path).formatted_resume


def load_resume_profile(path: str) -> ResumeProfile:
    """Load a resume JSON file and return resume text plus interview context."""
    resume_path = Path(path)
    data = json.loads(resume_path.read_text(encoding="utf-8"))
    interview_context = str(
        data.get("interview_context", DEFAULT_INTERVIEW_CONTEXT)
    ).strip() or DEFAULT_INTERVIEW_CONTEXT
    return ResumeProfile(
        filename=resume_path.name,
        company_name=_resume_company_name_from_filename(resume_path.stem),
        formatted_resume=format_resume(data),
        interview_context=interview_context,
    )


def _resume_company_name_from_filename(stem: str) -> str:
    normalized = stem.strip().lower()
    overrides = {
        "betterment": "Betterment",
        "bilt": "Bilt",
        "mta": "MTA",
        "default": "Default",
    }
    if normalized in overrides:
        return overrides[normalized]
    return stem.replace("_", " ").replace("-", " ").strip().title() or "Unknown"


def format_resume(data: dict) -> str:
    """Convert resume dict to structured plain text for LLM injection."""
    lines: list[str] = ["---"]

    name = str(data.get("name", "")).strip()
    title = str(data.get("title", "")).strip()
    if name or title:
        if name:
            lines.append(f"Name: {name}")
        if title:
            lines.append(f"Title: {title}")
        lines.append("")

    summary = str(data.get("summary", "")).strip()
    if summary:
        lines.append("Summary:")
        lines.append(summary)
        lines.append("")

    experience_items = []
    for item in data.get("experience", []):
        if not isinstance(item, dict):
            continue
        company = str(item.get("company", "")).strip()
        role = str(item.get("title", "")).strip()
        duration = str(item.get("duration", "")).strip()
        bullets = [
            str(bullet).strip()
            for bullet in item.get("bullets", [])
            if str(bullet).strip()
        ]
        if not any([company, role, duration, bullets]):
            continue

        header_parts = [part for part in [company, role, duration] if part]
        if header_parts:
            experience_items.append(" | ".join(header_parts))
        for bullet in bullets:
            experience_items.append(f"- {bullet}")
        experience_items.append("")

    if experience_items:
        lines.append("Experience:")
        lines.extend(experience_items[:-1] if experience_items[-1] == "" else experience_items)
        lines.append("")

    skills = data.get("skills", {})
    if isinstance(skills, dict):
        skill_lines = []
        for label, key in [
            ("Languages", "languages"),
            ("Frameworks", "frameworks"),
            ("Messaging", "messaging"),
            ("Databases", "databases"),
            ("Cloud", "cloud"),
            ("Tools", "tools"),
        ]:
            values = [
                str(value).strip()
                for value in skills.get(key, [])
                if str(value).strip()
            ]
            if values:
                skill_lines.append(f"{label}: {', '.join(values)}")

        if skill_lines:
            lines.append("Skills:")
            lines.extend(skill_lines)
            lines.append("")

    education_lines = []
    for item in data.get("education", []):
        if not isinstance(item, dict):
            continue
        school = str(item.get("school", "")).strip()
        degree = str(item.get("degree", "")).strip()
        year = str(item.get("year", "")).strip()
        if not any([school, degree, year]):
            continue
        education_lines.append(" | ".join(part for part in [school, degree, year] if part))

    if education_lines:
        lines.append("Education:")
        lines.extend(education_lines)
        lines.append("")

    if lines[-1] == "":
        lines.pop()
    lines.append("---")

    if lines == ["---", "---"]:
        return ""

    return "\n".join(lines)
