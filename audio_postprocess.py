from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


TOKEN_SYLLABLES = {
    "takukt": ("ta", "kukt"),
    "pakushk": ("pa", "kushk"),
    "kilost": ("ki", "lost"),
    "pulask": ("pu", "lask"),
    "palosk": ("pa", "losk"),
    "kazost": ("ka", "zost"),
    "djano": ("dja", "no"),
    "tnuka": ("tnu", "ka"),
    "plami": ("pla", "mi"),
    "pluna": ("plu", "na"),
    "plana": ("pla", "na"),
    "bluma": ("blu", "ma"),
    "dlamo": ("dla", "mo"),
    "kada": ("ka", "da"),
    "tika": ("ti", "ka"),
    "kika": ("ki", "ka"),
    "lumi": ("lu", "mi"),
    "nemo": ("ni", "mo"),
    "nimo": ("ni", "mo"),
    "la": ("la",),
    "ko": ("ko",),
    "tu": ("tu",),
    "to": ("to",),
    "tosh": ("tosh",),
    "mo": ("mo",),
    "na": ("na",),
    "po": ("po",),
    "ta": ("ta",),
    "su": ("su",),
    "za": ("za",),
    "tak": ("tak",),
    "kla": ("kla",),
}


@dataclass(frozen=True)
class EndingPreset:
    limiter_ceiling: float = 0.96
    frame_ms: float = 20.0
    hop_ms: float = 10.0
    tail_search_ms: float = 420.0
    min_tail_ms: float = 65.0
    final_start_gain_db: float = 0.4
    final_vowel_gain_db: float = 2.4
    final_end_gain_db: float = 2.0
    final_vowel_pitch_steps: float = 0.35
    crossfade_ms: float = 18.0


@dataclass(frozen=True)
class ProcessedAudio:
    wav_bytes: bytes
    sample_rate: int
    channels: int
    input_duration_seconds: float
    output_duration_seconds: float
    syllable_count: int
    final_syllable: str
    tail_adjusted: bool
    mime_type: str = "audio/wav"


def process_wav_file(
    input_path: str | Path,
    output_path: str | Path,
    syllable_text: str | None = None,
    preset: EndingPreset | None = None,
) -> ProcessedAudio:
    result = process_wav_bytes(Path(input_path).read_bytes(), syllable_text=syllable_text, preset=preset)
    Path(output_path).write_bytes(result.wav_bytes)
    return result


def process_wav_bytes(
    wav_bytes: bytes,
    syllable_text: str | None = None,
    preset: EndingPreset | None = None,
) -> ProcessedAudio:
    preset = preset or EndingPreset()
    audio, sample_rate, channels = _read_wav_bytes(wav_bytes)
    audio = _reduce_if_clipping(audio, preset.limiter_ceiling)
    syllables = _parse_syllables(syllable_text)
    final_syllable = syllables[-1] if syllables else ""

    processed, adjusted = _hold_final_syllable(audio, sample_rate, final_syllable, preset)
    processed = _soft_limiter(processed, preset.limiter_ceiling)

    return ProcessedAudio(
        wav_bytes=_write_wav_bytes(processed, sample_rate, channels),
        sample_rate=sample_rate,
        channels=channels,
        input_duration_seconds=len(audio) / sample_rate if sample_rate else 0.0,
        output_duration_seconds=len(processed) / sample_rate if sample_rate else 0.0,
        syllable_count=len(syllables),
        final_syllable=final_syllable,
        tail_adjusted=adjusted,
    )


def _hold_final_syllable(
    audio: np.ndarray,
    sample_rate: int,
    final_syllable: str,
    preset: EndingPreset,
) -> tuple[np.ndarray, bool]:
    tail = _find_final_voiced_tail(audio, sample_rate, preset)
    if tail is None:
        return audio.copy(), False

    tail_start, tail_end = tail
    vowel_start, vowel_end = _estimate_vowel_region(
        tail_start,
        tail_end,
        sample_rate,
        final_syllable,
    )
    out = audio.copy()
    out = _apply_final_intensity_hold(out, tail_start, vowel_start, vowel_end, tail_end, preset)
    out = _lift_vowel_pitch(out, sample_rate, vowel_start, vowel_end, preset)
    return out, True


def _find_final_voiced_tail(
    audio: np.ndarray,
    sample_rate: int,
    preset: EndingPreset,
) -> tuple[int, int] | None:
    mono = _to_mono(audio)
    if mono.size == 0:
        return None

    frame_len = max(1, int(sample_rate * preset.frame_ms / 1000))
    hop = max(1, int(sample_rate * preset.hop_ms / 1000))
    rms, starts = _frame_rms(mono, frame_len, hop)
    if rms.size == 0:
        return None

    dbfs = 20.0 * np.log10(np.maximum(rms, 1e-9))
    voiced_threshold = max(-48.0, float(np.median(dbfs)) - 24.0)
    voiced = dbfs > voiced_threshold
    voiced_indices = np.flatnonzero(voiced)
    if not voiced_indices.size:
        return None

    search_start = max(0, len(mono) - int(sample_rate * preset.tail_search_ms / 1000))
    candidates = [idx for idx in voiced_indices if starts[idx] >= search_start]
    if not candidates:
        candidates = voiced_indices[-8:].tolist()

    last = int(candidates[-1])
    first = last
    while first > 0 and voiced[first - 1] and starts[first] >= search_start:
        first -= 1

    tail_start = max(0, int(starts[first]) - int(sample_rate * 0.012))
    tail_end = min(len(mono), int(starts[last]) + frame_len)
    min_tail = int(sample_rate * preset.min_tail_ms / 1000)
    if tail_end - tail_start < min_tail:
        tail_start = max(0, tail_end - min_tail)
    return tail_start, tail_end


def _estimate_vowel_region(
    tail_start: int,
    tail_end: int,
    sample_rate: int,
    final_syllable: str,
) -> tuple[int, int]:
    onset, vowel, coda = _split_syllable(final_syllable)
    tail_len = max(1, tail_end - tail_start)
    coda_len = 0
    if coda:
        coda_len = min(int(sample_rate * 0.075), max(int(tail_len * 0.18), int(sample_rate * 0.025)))
    onset_len = 0
    if onset:
        onset_len = min(int(sample_rate * 0.055), max(int(tail_len * 0.12), int(sample_rate * 0.015)))

    vowel_start = min(tail_end - 1, tail_start + onset_len)
    vowel_end = max(vowel_start + 1, tail_end - coda_len)

    min_vowel = int(sample_rate * 0.045)
    if vowel and vowel_end - vowel_start < min_vowel:
        vowel_start = max(tail_start, vowel_end - min_vowel)
    return vowel_start, vowel_end


def _split_syllable(syllable: str) -> tuple[str, str, str]:
    syllable = (syllable or "").lower()
    match = re.search(r"[aeiou]+", syllable)
    if not match:
        return "", "", syllable
    start, end = match.span()
    if syllable[start:end] == "i" and end < len(syllable) and syllable[end] == "e":
        end += 1
    return syllable[:start], syllable[start:end], syllable[end:]


def _apply_final_intensity_hold(
    audio: np.ndarray,
    tail_start: int,
    vowel_start: int,
    vowel_end: int,
    tail_end: int,
    preset: EndingPreset,
) -> np.ndarray:
    out = audio.copy()
    points = np.array([tail_start, vowel_start, vowel_end, tail_end - 1], dtype=np.float32)
    gains = np.array(
        [
            preset.final_start_gain_db,
            preset.final_vowel_gain_db,
            preset.final_vowel_gain_db,
            preset.final_end_gain_db,
        ],
        dtype=np.float32,
    )
    points, unique_indices = np.unique(points, return_index=True)
    gains = gains[unique_indices]
    if len(points) < 2:
        return out

    envelope = np.zeros(len(out), dtype=np.float32)
    span = np.arange(tail_start, tail_end, dtype=np.float32)
    envelope[tail_start:tail_end] = np.interp(span, points, gains)
    smooth_width = max(5, min(301, (tail_end - tail_start) // 4))
    envelope[tail_start:tail_end] = _smooth_1d(envelope[tail_start:tail_end], smooth_width)
    out *= (10.0 ** (envelope / 20.0)).reshape(-1, 1)
    return out


def _lift_vowel_pitch(
    audio: np.ndarray,
    sample_rate: int,
    vowel_start: int,
    vowel_end: int,
    preset: EndingPreset,
) -> np.ndarray:
    if preset.final_vowel_pitch_steps <= 0.0:
        return audio
    if vowel_end - vowel_start < int(sample_rate * 0.055):
        return audio

    segment = audio[vowel_start:vowel_end]
    shifted = _pitch_shift(segment, sample_rate, preset.final_vowel_pitch_steps)
    fade = min(int(sample_rate * preset.crossfade_ms / 1000), max(1, len(segment) // 3))
    window = np.ones(len(segment), dtype=np.float32)
    if fade > 1:
        window[:fade] = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        window[-fade:] = np.linspace(1.0, 0.0, fade, dtype=np.float32)
    out = audio.copy()
    out[vowel_start:vowel_end] = segment * (1.0 - window.reshape(-1, 1)) + shifted * window.reshape(-1, 1)
    return out


def _pitch_shift(audio: np.ndarray, sample_rate: int, steps: float) -> np.ndarray:
    if not audio.size or abs(steps) < 0.05:
        return audio.copy()
    channels = []
    for channel in range(audio.shape[1]):
        try:
            shifted = librosa.effects.pitch_shift(
                audio[:, channel].astype(np.float32),
                sr=sample_rate,
                n_steps=steps,
            )
        except Exception:
            shifted = audio[:, channel].copy()
        channels.append(_fit_vector_length(shifted, len(audio)))
    return np.stack(channels, axis=1).astype(np.float32)


def _parse_syllables(text: str | None) -> tuple[str, ...]:
    if not text:
        return ()
    text = re.sub(r"\[[^\]]+\]", " ", text.lower())
    syllables: list[str] = []
    for token in re.findall(r"[a-z:]+", text):
        syllables.extend(TOKEN_SYLLABLES.get(token, _rough_syllables_from_token(token)))
    return tuple(syllable for syllable in syllables if syllable)


def _rough_syllables_from_token(token: str) -> tuple[str, ...]:
    token = token.replace(":", "")
    if not token:
        return ()
    vowels = set("aeiou")
    syllables: list[str] = []
    start = 0
    index = 0
    while index < len(token):
        while index < len(token) and token[index] not in vowels:
            index += 1
        if index >= len(token):
            break
        index += 1
        if index < len(token) and token[index - 1 : index + 1] == "ie":
            index += 1
        while index < len(token) and token[index] == token[index - 1]:
            index += 1
        syllables.append(token[start:index])
        start = index
    if not syllables:
        return (token,)
    if start < len(token):
        syllables[-1] += token[start:]
    return tuple(syllables)


def _frame_rms(mono: np.ndarray, frame_len: int, hop: int) -> tuple[np.ndarray, np.ndarray]:
    if mono.size == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int64)
    padded = mono
    if len(padded) < frame_len:
        padded = np.pad(padded, (0, frame_len - len(padded)))
    starts = np.arange(0, max(1, len(padded) - frame_len + 1), hop, dtype=np.int64)
    if starts[-1] + frame_len < len(padded):
        starts = np.append(starts, len(padded) - frame_len)
    rms = np.empty(len(starts), dtype=np.float32)
    for i, start in enumerate(starts):
        frame = padded[start : start + frame_len]
        rms[i] = float(np.sqrt(np.mean(frame * frame) + 1e-12))
    return rms, starts


def _fit_vector_length(values: np.ndarray, target_len: int) -> np.ndarray:
    if len(values) == target_len:
        return values.astype(np.float32)
    if len(values) <= 1:
        return np.zeros(target_len, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    return np.interp(x_new, x_old, values).astype(np.float32)


def _smooth_1d(values: np.ndarray, width: int) -> np.ndarray:
    if len(values) < width or width <= 1:
        return values.astype(np.float32)
    if width % 2 == 0:
        width += 1
    kernel = np.ones(width, dtype=np.float32) / width
    return np.convolve(values, kernel, mode="same").astype(np.float32)


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32)
    return np.mean(audio, axis=1).astype(np.float32)


def _reduce_if_clipping(audio: np.ndarray, ceiling: float) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= ceiling or peak <= 1e-9:
        return audio.astype(np.float32)
    return (audio * (ceiling / peak)).astype(np.float32)


def _soft_limiter(audio: np.ndarray, ceiling: float = 0.96) -> np.ndarray:
    if not audio.size:
        return audio
    limited = np.tanh(audio * 1.15) / np.tanh(1.15)
    peak = float(np.max(np.abs(limited)))
    if peak > ceiling:
        limited = limited * (ceiling / peak)
    return np.clip(limited, -ceiling, ceiling).astype(np.float32)


def _read_wav_bytes(wav_bytes: bytes) -> tuple[np.ndarray, int, int]:
    with io.BytesIO(wav_bytes) as buffer:
        audio, sample_rate = sf.read(buffer, always_2d=True, dtype="float32")
    return audio.astype(np.float32), int(sample_rate), int(audio.shape[1])


def _write_wav_bytes(audio: np.ndarray, sample_rate: int, channels: int) -> bytes:
    output = io.BytesIO()
    data = audio[:, 0] if channels == 1 and audio.ndim == 2 else audio
    sf.write(output, data, sample_rate, format="WAV", subtype="PCM_16")
    return output.getvalue()
