"""Shared helper for calling Sarvam AI's APIs.

Every stage of the pipeline (script writer, TTS, STT, translation) talks to
Sarvam through this file, so the API key handling and error handling live in
one place.
"""

import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()  # reads SARVAM_API_KEY from the .env file into the environment

BASE_URL = "https://api.sarvam.ai"
# The plain "sarvam-105b" model spends 10,000+ hidden reasoning tokens on a
# one-page story (about 70 seconds per call). The conversations variant answers
# directly in about 3 seconds, and its output is just as good for this task.
CHAT_MODEL = "sarvam-105b-conversations"


class SarvamError(RuntimeError):
    """Raised when Sarvam returns an error we cannot recover from."""


class OutOfTokens(SarvamError):
    """Raised when the model hit max_tokens before finishing its answer."""


def _headers() -> dict:
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise SarvamError(
            "SARVAM_API_KEY is not set. Copy .env.example to .env and paste your key."
        )
    return {"api-subscription-key": api_key, "Authorization": f"Bearer {api_key}"}


def post(
    path: str,
    payload: dict,
    retries: int = 3,
    timeout: int = 120,
    files: dict | None = None,
) -> dict:
    """POST to a Sarvam endpoint and return the parsed JSON response.

    payload is sent as JSON, or as form fields when files are attached (the
    speech-to-text endpoint takes an uploaded audio file).

    Rate limits (429) and server errors (5xx) are temporary, so we wait and
    try again. Anything else (bad key, bad request) fails immediately because
    retrying would not help.
    """
    url = f"{BASE_URL}{path}"
    for attempt in range(1, retries + 1):
        if files:
            response = requests.post(
                url, data=payload, files=files, headers=_headers(), timeout=timeout
            )
        else:
            response = requests.post(url, json=payload, headers=_headers(), timeout=timeout)
        if response.ok:
            return response.json()
        temporary = response.status_code == 429 or response.status_code >= 500
        if temporary and attempt < retries:
            time.sleep(2**attempt)  # 2s, then 4s
            continue
        raise SarvamError(f"{path} failed with HTTP {response.status_code}: {response.text}")
    raise SarvamError(f"{path} failed after {retries} attempts")


def chat(messages: list[dict], response_format: dict | None = None, **options) -> str:
    """Send a conversation to Sarvam's chat model and return the reply text."""
    payload = {"model": CHAT_MODEL, "messages": messages, **options}
    if response_format:
        payload["response_format"] = response_format

    data = post("/v1/chat/completions", payload)
    choice = data["choices"][0]
    if choice["finish_reason"] == "length":
        raise OutOfTokens(
            "The model ran out of tokens before finishing. "
            "Use a shorter story or raise max_tokens."
        )
    content = choice["message"].get("content")
    if not content:
        raise SarvamError(f"The model returned no text: {choice['message']}")
    return content
