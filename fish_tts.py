from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


FISH_TTS_URL = "https://api.fish.audio/v1/tts"
FISH_TTS_TOKEN_OVERRIDES = {
    "nemo": "nimo",
}


@dataclass
class FishTTSResult:
    audio_bytes: bytes
    mime_type: str


def nemo_to_fish_tts_text(nemo_text: str) -> str:
    parts: list[str] = []
    for raw_token in (nemo_text or "").split():
        token = raw_token.strip()
        if not token:
            continue
        parts.append(FISH_TTS_TOKEN_OVERRIDES.get(token.lower(), token))
    return " ".join(parts)


def synthesize_fish_tts(
    text: str,
    api_key: str,
    reference_id: str,
    model: str = "s2-pro",
    speed: float = 1.0,
    timeout: int = 120,
) -> FishTTSResult:
    text = (text or "").strip()
    api_key = (api_key or "").strip()
    reference_id = (reference_id or "").strip()
    model = (model or "s2-pro").strip()

    if not text:
        raise ValueError("TTS text is empty.")
    if not api_key:
        raise ValueError("FISH_API_KEY is not configured.")
    if not reference_id:
        raise ValueError("FISH_REFERENCE_ID or FISH_SPEAKER_ID is not configured.")

    payload = {
        "text": text,
        "reference_id": reference_id,
        "temperature": 0.7,
        "top_p": 0.7,
        "prosody": {
            "speed": _clamped_speed(speed),
            "volume": 0,
            "normalize_loudness": True,
        },
        "chunk_length": 200,
        "normalize": False,
        "format": "mp3",
        "sample_rate": 44100,
        "mp3_bitrate": 128,
        "latency": "normal",
        "max_new_tokens": 1024,
        "repetition_penalty": 1.2,
        "min_chunk_length": 50,
        "condition_on_previous_chunks": True,
        "early_stop_threshold": 1,
    }
    request = Request(
        FISH_TTS_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "model": model,
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            mime_type = response.headers.get("Content-Type") or "audio/mpeg"
            return FishTTSResult(audio_bytes=response.read(), mime_type=mime_type.split(";")[0])
    except HTTPError as exc:
        raise RuntimeError(_fish_error_message(exc)) from exc
    except URLError as exc:
        raise RuntimeError(f"Could not connect to Fish Audio: {exc.reason}") from exc


def _clamped_speed(speed: float) -> float:
    return max(0.5, min(float(speed or 1.0), 1.8))


def _fish_error_message(exc: HTTPError) -> str:
    raw = exc.read().decode("utf-8", errors="replace")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = {}
    message = payload.get("message") or raw or exc.reason
    return f"Fish Audio TTS failed ({exc.code}): {message}"
