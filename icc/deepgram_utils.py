from __future__ import annotations

import logging
from typing import Any

import requests


logger = logging.getLogger(__name__)

DEEPGRAM_PROJECTS_URL = "https://api.deepgram.com/v1/projects"
DEEPGRAM_BALANCES_URL_TEMPLATE = "https://api.deepgram.com/v1/projects/{project_id}/balances"


def check_deepgram_balance(
    api_key: str,
    warning_threshold: float = 1.0,
    timeout_seconds: float = 10.0,
) -> None:
    if not api_key:
        logger.warning("Deepgram balance check skipped: DEEPGRAM_API_KEY is not set.")
        return

    headers = {"Authorization": f"Token {api_key}"}

    try:
        projects = _fetch_projects(headers=headers, timeout_seconds=timeout_seconds)

        if not projects:
            logger.warning("Deepgram balance check failed: no projects were returned.")
            return

        project_id = str(projects[0].get("project_id") or projects[0].get("id") or "").strip()
        if not project_id:
            logger.warning("Deepgram balance check failed: first project is missing an id.")
            return

        balances_response = requests.get(
            DEEPGRAM_BALANCES_URL_TEMPLATE.format(project_id=project_id),
            headers=headers,
            timeout=timeout_seconds,
        )
        if balances_response.status_code == 403:
            logger.info(
                "\u2139\ufe0f  Deepgram API key is valid "
                "(balance check not permitted for this key type). "
                "Check usage at: https://console.deepgram.com"
            )
            return
        balances_response.raise_for_status()
        balances_payload = balances_response.json()
        balances = _extract_balances(balances_payload)

        if not balances:
            logger.warning("Deepgram balance check failed: no balances were returned.")
            return

        for balance in balances:
            amount = _parse_amount(balance)
            currency = _parse_units(balance)

            logger.info("\u2705 Deepgram balance: $%.2f (%s)", amount, currency)
            if amount < warning_threshold:
                logger.warning(
                    "\u26a0\ufe0f  Deepgram balance low: $%.2f \u2014 consider topping up!",
                    amount,
                )
    except requests.RequestException as exc:
        logger.warning("Deepgram balance check failed: %s", exc)
    except (TypeError, ValueError, KeyError) as exc:
        logger.warning("Deepgram balance check returned unexpected data: %s", exc)


def _fetch_projects(headers: dict[str, str], timeout_seconds: float) -> list[dict[str, Any]]:
    projects_response = requests.get(
        DEEPGRAM_PROJECTS_URL,
        headers=headers,
        timeout=timeout_seconds,
    )
    projects_response.raise_for_status()
    projects_payload = projects_response.json()
    return _extract_projects(projects_payload)


def _extract_projects(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        projects = payload.get("projects", [])
        if isinstance(projects, list):
            return [item for item in projects if isinstance(item, dict)]
    return []


def _extract_balances(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        balances = payload.get("balances", [])
        if isinstance(balances, list):
            return [item for item in balances if isinstance(item, dict)]
    return []


def _parse_amount(balance: dict[str, Any]) -> float:
    raw_amount = balance.get("amount")
    if raw_amount is None:
        raw_amount = balance.get("value", 0)
    return float(raw_amount)


def _parse_units(balance: dict[str, Any]) -> str:
    for key in ("units", "unit", "currency"):
        value = balance.get(key)
        if value:
            return str(value).upper()
    return "USD"
