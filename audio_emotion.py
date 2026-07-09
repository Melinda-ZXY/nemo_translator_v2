from __future__ import annotations

import io
import re
from dataclasses import dataclass, replace
from pathlib import Path
from collections.abc import Iterable
from typing import Literal

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt


EmotionMode = Literal["normal", "urgent", "angry", "angry_urgent"]


@dataclass(frozen=True)
class EmotionPreset:
    mode: EmotionMode
    target_dbfs: float = -20.0
    limiter_ceiling: float = 0.96
    frame_ms: float = 20.0
    hop_ms: float = 10.0
    phrase_pause_ms: float = 140.0
    boundary_pad_ms: float = 20.0
    crossfade_ms: float = 28.0
    onset_rise_db: float = 5.5
    min_feature_distance_ms: float = 85.0
    max_peak_count: int = 4
    urgent_speed: float = 1.18
    urgent_pitch_steps: float = 0.0
    urgent_gain_db: float = 0.7
    urgent_presence_db: float = 0.4
    angry_weak_speed: float = 1.06
    angry_strong_speed: float = 1.0
    angry_baseline_pitch_steps: float = 0.0
    angry_peak_pitch_steps: float = 0.0
    angry_peak_gain_db: float = 1.8
    angry_onset_gain_db: float = 1.2
    angry_presence_db: float = 0.6
    protect_window_ms: float = 80.0
    pitch_spike_window_ms: float = 45.0
    tail_search_ms: float = 360.0
    tail_min_ms: float = 70.0
    tail_stretch: float = 1.10
    tail_max_add_ms: float = 160.0
    tail_fall_steps: float = 0.0
    tail_start_gain_db: float = 1.6
    tail_end_gain_db: float = -3.2
    angry_urgent_begin_fraction: float = 0.45
    angry_urgent_begin_speed: float = 1.18
    angry_urgent_begin_pitch_steps: float = 0.0
    angry_urgent_middle_speed: float = 1.06


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
class PhraseRegion:
    start: int
    end: int
    peaks: tuple[int, ...]
    onsets: tuple[int, ...]
    tail_start: int
    tail_end: int


@dataclass(frozen=True)
class FrameAnalysis:
    dbfs: np.ndarray
    rms: np.ndarray
    silence: np.ndarray
    threshold_dbfs: float
    frame_starts: np.ndarray
    frame_len: int
    hop: int
    phrases: tuple[PhraseRegion, ...]


@dataclass(frozen=True)
class ProcessedAudio:
    wav_bytes: bytes
    sample_rate: int
    channels: int
    mode: EmotionMode
    phrase_count: int
    syllable_count: int
    threshold_dbfs: float
    input_duration_seconds: float
    output_duration_seconds: float
    mime_type: str = "audio/wav"


EMOTION_PRESETS: dict[EmotionMode, EmotionPreset] = {
    "normal": EmotionPreset(mode="normal", phrase_pause_ms=160.0),
    "urgent": EmotionPreset(
        mode="urgent",
        phrase_pause_ms=95.0,
        crossfade_ms=24.0,
        urgent_speed=1.18,
        urgent_pitch_steps=0.0,
        urgent_gain_db=0.7,
        urgent_presence_db=0.4,
    ),
    "angry": EmotionPreset(
        mode="angry",
        phrase_pause_ms=135.0,
        crossfade_ms=30.0,
        angry_weak_speed=1.06,
        angry_strong_speed=1.0,
        angry_baseline_pitch_steps=0.0,
        angry_peak_pitch_steps=0.0,
        angry_peak_gain_db=1.8,
        angry_onset_gain_db=1.2,
        angry_presence_db=0.6,
        pitch_spike_window_ms=45.0,
        tail_stretch=1.10,
        tail_fall_steps=0.0,
    ),
    "angry_urgent": EmotionPreset(
        mode="angry_urgent",
        phrase_pause_ms=85.0,
        crossfade_ms=28.0,
        urgent_speed=1.18,
        urgent_pitch_steps=0.0,
        angry_peak_pitch_steps=0.0,
        angry_peak_gain_db=1.8,
        angry_onset_gain_db=1.2,
        angry_presence_db=0.5,
        pitch_spike_window_ms=45.0,
        tail_stretch=1.08,
        tail_fall_steps=0.0,
        angry_urgent_begin_speed=1.18,
        angry_urgent_begin_pitch_steps=0.0,
        angry_urgent_middle_speed=1.06,
    ),
}


def get_emotion_preset(mode: EmotionMode = "normal", **overrides: float | int) -> EmotionPreset:
    preset = EMOTION_PRESETS[mode]
    return replace(preset, **overrides) if overrides else preset


def process_wav_file(
    input_path: str | Path,
    output_path: str | Path,
    mode: EmotionMode = "normal",
    preset: EmotionPreset | None = None,
    syllable_text: str | None = None,
) -> ProcessedAudio:
    wav_bytes = Path(input_path).read_bytes()
    result = process_wav_bytes(wav_bytes, mode=mode, preset=preset, syllable_text=syllable_text)
    Path(output_path).write_bytes(result.wav_bytes)
    return result


def process_wav_bytes(
    wav_bytes: bytes,
    mode: EmotionMode = "normal",
    preset: EmotionPreset | None = None,
    syllable_text: str | None = None,
) -> ProcessedAudio:
    preset = preset or EMOTION_PRESETS[mode]
    mode = preset.mode
    audio, sample_rate, channels = _read_wav_bytes(wav_bytes)
    audio = _loudness_normalize(audio, preset.target_dbfs, preset.limiter_ceiling)
    analysis = analyze_waveform(audio, sample_rate, preset)
    syllables = _parse_syllables(syllable_text)

    if mode == "normal" or not analysis.phrases:
        processed = _soft_limiter(audio, preset.limiter_ceiling)
    else:
        syllable_counts = _assign_syllable_counts(analysis, syllables)
        processed = _render_processed_audio(audio, sample_rate, preset, analysis, syllable_counts)
        processed = _match_rms(processed, audio, max_gain_db=7.0)
        processed = _soft_limiter(processed, preset.limiter_ceiling)

    return ProcessedAudio(
        wav_bytes=_write_wav_bytes(processed, sample_rate, channels),
        sample_rate=sample_rate,
        channels=channels,
        mode=mode,
        phrase_count=len(analysis.phrases),
        syllable_count=len(syllables),
        threshold_dbfs=analysis.threshold_dbfs,
        input_duration_seconds=len(audio) / sample_rate if sample_rate else 0.0,
        output_duration_seconds=len(processed) / sample_rate if sample_rate else 0.0,
    )


def analyze_waveform(audio: np.ndarray, sample_rate: int, preset: EmotionPreset) -> FrameAnalysis:
    mono = _to_mono(audio)
    frame_len = max(1, int(sample_rate * preset.frame_ms / 1000))
    hop = max(1, int(sample_rate * preset.hop_ms / 1000))
    rms, frame_starts = _frame_rms(mono, frame_len, hop)
    dbfs = 20.0 * np.log10(np.maximum(rms, 1e-9))
    median_dbfs = float(np.median(dbfs)) if dbfs.size else -90.0
    threshold_dbfs = max(-45.0, median_dbfs - 25.0)
    silence = dbfs < threshold_dbfs
    phrases = _detect_phrases(dbfs, silence, frame_starts, frame_len, hop, len(mono), sample_rate, preset)
    return FrameAnalysis(
        dbfs=dbfs,
        rms=rms,
        silence=silence,
        threshold_dbfs=threshold_dbfs,
        frame_starts=frame_starts,
        frame_len=frame_len,
        hop=hop,
        phrases=phrases,
    )


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


def _assign_syllable_counts(analysis: FrameAnalysis, syllables: tuple[str, ...]) -> tuple[int, ...]:
    if not analysis.phrases:
        return ()
    if not syllables:
        return tuple(0 for _ in analysis.phrases)

    durations = np.array([max(1, phrase.end - phrase.start) for phrase in analysis.phrases], dtype=np.float64)
    weights = durations / max(1.0, float(np.sum(durations)))
    raw = weights * len(syllables)
    counts = np.maximum(1, np.floor(raw).astype(int))
    while int(np.sum(counts)) < len(syllables):
        counts[int(np.argmax(raw - counts))] += 1
    while int(np.sum(counts)) > len(syllables) and np.max(counts) > 1:
        counts[int(np.argmax(counts - raw))] -= 1
    return tuple(int(count) for count in counts)


def _render_processed_audio(
    audio: np.ndarray,
    sample_rate: int,
    preset: EmotionPreset,
    analysis: FrameAnalysis,
    syllable_counts: tuple[int, ...],
) -> np.ndarray:
    parts: list[np.ndarray] = []
    cursor = 0
    for phrase_index, phrase in enumerate(analysis.phrases):
        if phrase.start > cursor:
            parts.append(_compress_pause(audio[cursor : phrase.start], sample_rate, preset))
        phrase_audio = audio[phrase.start : phrase.end]
        syllable_count = syllable_counts[phrase_index] if phrase_index < len(syllable_counts) else 0
        parts.append(_process_phrase(phrase_audio, phrase, sample_rate, preset, syllable_count))
        cursor = phrase.end

    if cursor < len(audio):
        parts.append(_compress_pause(audio[cursor:], sample_rate, preset))

    return _concat_with_crossfades(parts, sample_rate, preset.crossfade_ms)


def _process_phrase(
    phrase_audio: np.ndarray,
    phrase: PhraseRegion,
    sample_rate: int,
    preset: EmotionPreset,
    syllable_count: int = 0,
) -> np.ndarray:
    if preset.mode == "urgent":
        return _process_urgent_phrase(phrase_audio, sample_rate, preset, syllable_count)
    if preset.mode == "angry":
        return _process_angry_phrase(phrase_audio, phrase, sample_rate, preset, syllable_count)
    if preset.mode == "angry_urgent":
        return _process_angry_urgent_phrase(phrase_audio, phrase, sample_rate, preset, syllable_count)
    return phrase_audio


def _process_urgent_phrase(
    audio: np.ndarray,
    sample_rate: int,
    preset: EmotionPreset,
    syllable_count: int = 0,
) -> np.ndarray:
    out = _syllable_time_stretch(audio, sample_rate, syllable_count, preset.urgent_speed, preset.crossfade_ms)
    out = _pitch_shift(out, sample_rate, preset.urgent_pitch_steps)
    out = _apply_gain_db(out, preset.urgent_gain_db)
    out = _apply_syllable_gain_contour(out, syllable_count, start_db=0.2, mid_db=0.7, end_db=0.4)
    return _presence_boost(out, sample_rate, preset.urgent_presence_db, low_hz=2000, high_hz=4000)


def _process_angry_phrase(
    audio: np.ndarray,
    phrase: PhraseRegion,
    sample_rate: int,
    preset: EmotionPreset,
    syllable_count: int = 0,
) -> np.ndarray:
    body, tail, body_len = _split_final_tail(audio, phrase, sample_rate, preset)
    if body.size:
        body = _pitch_shift(body, sample_rate, preset.angry_baseline_pitch_steps)
        body_syllables = _body_syllable_count(syllable_count, len(body), len(audio))
        rates = _angry_syllable_rates(body_syllables, preset)
        body = _syllable_time_stretch(body, sample_rate, body_syllables, rates, preset.crossfade_ms)
        body = _apply_syllable_gain_contour(
            body,
            body_syllables,
            start_db=0.7,
            mid_db=preset.angry_peak_gain_db,
            end_db=preset.angry_onset_gain_db,
        )
        body = _smooth_phrase_onset(body, sample_rate, preset.angry_onset_gain_db, duration_ms=85.0)
        body = _presence_boost(body, sample_rate, preset.angry_presence_db, low_hz=2000, high_hz=5000)

    if tail.size:
        tail = _process_angry_tail(tail, sample_rate, preset)

    return _concat_with_crossfades([body, tail], sample_rate, preset.crossfade_ms)


def _process_angry_urgent_phrase(
    audio: np.ndarray,
    phrase: PhraseRegion,
    sample_rate: int,
    preset: EmotionPreset,
    syllable_count: int = 0,
) -> np.ndarray:
    body, tail, _ = _split_final_tail(audio, phrase, sample_rate, preset)
    if body.size:
        body_syllables = _body_syllable_count(syllable_count, len(body), len(audio))
        rates = _angry_urgent_syllable_rates(body_syllables, preset)
        body = _syllable_time_stretch(body, sample_rate, body_syllables, rates, preset.crossfade_ms)
        body = _apply_syllable_gain_contour(
            body,
            body_syllables,
            start_db=0.5,
            mid_db=1.1,
            end_db=preset.angry_peak_gain_db,
        )
        body = _smooth_phrase_onset(body, sample_rate, preset.urgent_gain_db, duration_ms=90.0)
        body = _presence_boost(body, sample_rate, preset.angry_presence_db, low_hz=2000, high_hz=5000)

    if tail.size:
        tail = _process_angry_tail(tail, sample_rate, preset)

    return _concat_with_crossfades([body, tail], sample_rate, preset.crossfade_ms)


def _split_final_tail(
    audio: np.ndarray,
    phrase: PhraseRegion,
    sample_rate: int,
    preset: EmotionPreset,
) -> tuple[np.ndarray, np.ndarray, int]:
    tail_start = max(0, phrase.tail_start - phrase.start)
    tail_end = max(0, phrase.tail_end - phrase.start)
    min_tail = int(sample_rate * preset.tail_min_ms / 1000)
    if tail_end <= tail_start or len(audio) - tail_start < min_tail:
        return audio, audio[:0], len(audio)
    tail_start = max(0, min(tail_start, len(audio) - min_tail))
    return audio[:tail_start], audio[tail_start:], tail_start


def _process_angry_tail(audio: np.ndarray, sample_rate: int, preset: EmotionPreset) -> np.ndarray:
    if len(audio) < int(sample_rate * 0.04):
        return audio

    max_len = len(audio) + int(sample_rate * preset.tail_max_add_ms / 1000)
    stretch = min(preset.tail_stretch, max_len / max(1, len(audio)))
    out = _time_stretch(audio, sample_rate, 1.0 / stretch)
    out = _falling_pitch_contour(out, sample_rate, preset.tail_fall_steps)
    out = _tail_intensity_contour(out, preset.tail_start_gain_db, preset.tail_end_gain_db)
    return _presence_boost(out, sample_rate, preset.angry_presence_db, low_hz=2000, high_hz=5000)


def _compress_pause(audio: np.ndarray, sample_rate: int, preset: EmotionPreset) -> np.ndarray:
    if not audio.size or preset.mode == "normal":
        return audio
    original_ms = 1000.0 * len(audio) / sample_rate
    target_ms = _target_pause_ms(original_ms, preset.mode)
    target_len = int(sample_rate * target_ms / 1000)
    if target_len >= len(audio):
        return audio
    return _fit_length(audio, max(0, target_len))


def _target_pause_ms(original_ms: float, mode: EmotionMode) -> float:
    if mode == "urgent":
        if original_ms <= 80.0:
            return original_ms
        if original_ms >= 600.0:
            return 70.0
        return float(np.clip(original_ms * 0.32, 20.0, 50.0))
    if mode == "angry":
        if original_ms <= 120.0:
            return original_ms
        if original_ms >= 700.0:
            return 150.0
        return float(np.clip(original_ms * 0.55, 60.0, 100.0))
    if mode == "angry_urgent":
        if original_ms <= 70.0:
            return original_ms
        if original_ms >= 600.0:
            return 75.0
        return float(np.clip(original_ms * 0.32, 20.0, 50.0))
    return original_ms


def _detect_phrases(
    dbfs: np.ndarray,
    silence: np.ndarray,
    frame_starts: np.ndarray,
    frame_len: int,
    hop: int,
    total_samples: int,
    sample_rate: int,
    preset: EmotionPreset,
) -> tuple[PhraseRegion, ...]:
    speech_frames = np.flatnonzero(~silence)
    if not speech_frames.size:
        return ()

    pad = int(sample_rate * preset.boundary_pad_ms / 1000)
    min_phrase = int(sample_rate * 45 / 1000)
    long_silent_runs = [
        run
        for run in _runs(silence)
        if run[1] > run[0] and (run[1] - run[0]) * hop >= int(sample_rate * preset.phrase_pause_ms / 1000)
    ]

    phrase_ranges: list[tuple[int, int]] = []
    start_frame = int(speech_frames[0])
    current_start = max(0, int(frame_starts[start_frame]) - pad)
    last_speech_frame = int(speech_frames[-1])

    for silent_start, silent_end in long_silent_runs:
        if silent_end <= start_frame or silent_start >= last_speech_frame:
            continue
        phrase_end = min(total_samples, int(frame_starts[silent_start]) + pad)
        if phrase_end - current_start >= min_phrase:
            phrase_ranges.append((current_start, phrase_end))
        next_start = min(silent_end, len(frame_starts) - 1)
        current_start = max(0, int(frame_starts[next_start]) - pad)

    final_end = min(total_samples, int(frame_starts[last_speech_frame]) + frame_len + pad)
    if final_end - current_start >= min_phrase:
        phrase_ranges.append((current_start, final_end))

    return tuple(
        _with_phrase_features(start, end, dbfs, silence, frame_starts, frame_len, sample_rate, preset)
        for start, end in phrase_ranges
    )


def _with_phrase_features(
    start: int,
    end: int,
    dbfs: np.ndarray,
    silence: np.ndarray,
    frame_starts: np.ndarray,
    frame_len: int,
    sample_rate: int,
    preset: EmotionPreset,
) -> PhraseRegion:
    frame_centers = frame_starts + frame_len // 2
    mask = (frame_centers >= start) & (frame_centers <= end)
    indices = np.flatnonzero(mask)
    if not indices.size:
        return PhraseRegion(start=start, end=end, peaks=(), onsets=(start,), tail_start=max(start, end), tail_end=end)

    local_db = dbfs[indices]
    smooth_db = _smooth_1d(local_db, width=5)
    threshold = max(float(np.percentile(smooth_db, 75)), float(np.median(smooth_db) + 3.0))
    candidate_peaks = [
        int(frame_centers[indices[i]])
        for i in range(1, len(smooth_db) - 1)
        if smooth_db[i] >= threshold and smooth_db[i] >= smooth_db[i - 1] and smooth_db[i] >= smooth_db[i + 1]
    ]
    peak_scores = [float(dbfs[np.searchsorted(frame_centers, peak)]) for peak in candidate_peaks]
    peaks = _select_positions(candidate_peaks, peak_scores, sample_rate, preset)

    diff = np.diff(smooth_db, prepend=smooth_db[0])
    candidate_onsets: list[int] = []
    onset_scores: list[float] = []
    for i in range(1, len(diff)):
        if diff[i] >= preset.onset_rise_db and smooth_db[i] > np.median(smooth_db):
            candidate_onsets.append(int(frame_centers[indices[i]]))
            onset_scores.append(float(diff[i]))
    onsets = (start, *_select_positions(candidate_onsets, onset_scores, sample_rate, preset))
    tail_start, tail_end = _find_tail(start, end, dbfs, silence, frame_starts, frame_len, sample_rate, preset)
    return PhraseRegion(start=start, end=end, peaks=peaks, onsets=onsets, tail_start=tail_start, tail_end=tail_end)


def _find_tail(
    start: int,
    end: int,
    dbfs: np.ndarray,
    silence: np.ndarray,
    frame_starts: np.ndarray,
    frame_len: int,
    sample_rate: int,
    preset: EmotionPreset,
) -> tuple[int, int]:
    frame_centers = frame_starts + frame_len // 2
    mask = (frame_centers >= start) & (frame_centers <= end)
    indices = np.flatnonzero(mask)
    voiced = [idx for idx in indices if not silence[idx]]
    if not voiced:
        return end, end
    last = voiced[-1]
    first_allowed_sample = max(start, end - int(sample_rate * preset.tail_search_ms / 1000))
    first_allowed_frame = int(np.searchsorted(frame_starts, first_allowed_sample, side="left"))
    tail_first = last
    while tail_first > first_allowed_frame and tail_first - 1 in voiced:
        tail_first -= 1
    tail_end = min(end, int(frame_starts[last]) + frame_len)
    tail_start = max(first_allowed_sample, int(frame_starts[tail_first]) - int(sample_rate * 10 / 1000))
    if tail_end - tail_start < int(sample_rate * preset.tail_min_ms / 1000):
        tail_start = max(start, tail_end - int(sample_rate * preset.tail_min_ms / 1000))
    return tail_start, tail_end


def _local_features(audio: np.ndarray, sample_rate: int, preset: EmotionPreset) -> tuple[tuple[int, ...], tuple[int, ...]]:
    mono = _to_mono(audio)
    frame_len = max(1, int(sample_rate * preset.frame_ms / 1000))
    hop = max(1, int(sample_rate * preset.hop_ms / 1000))
    rms, frame_starts = _frame_rms(mono, frame_len, hop)
    dbfs = 20.0 * np.log10(np.maximum(rms, 1e-9))
    silence = dbfs < max(-45.0, float(np.median(dbfs)) - 25.0)
    phrase = _with_phrase_features(0, len(audio), dbfs, silence, frame_starts, frame_len, sample_rate, preset)
    return tuple(p - phrase.start for p in phrase.peaks), tuple(o - phrase.start for o in phrase.onsets)


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


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    if not mask.size:
        return []
    runs: list[tuple[int, int]] = []
    start = 0
    value = bool(mask[0])
    for idx in range(1, len(mask)):
        current = bool(mask[idx])
        if current != value:
            if value:
                runs.append((start, idx))
            start = idx
            value = current
    if value:
        runs.append((start, len(mask)))
    return runs


def _select_positions(
    positions: list[int],
    scores: list[float],
    sample_rate: int,
    preset: EmotionPreset,
) -> tuple[int, ...]:
    if not positions:
        return ()
    min_distance = int(sample_rate * preset.min_feature_distance_ms / 1000)
    ranked = sorted(zip(positions, scores, strict=False), key=lambda item: item[1], reverse=True)
    selected: list[int] = []
    for position, _score in ranked:
        if all(abs(position - existing) >= min_distance for existing in selected):
            selected.append(position)
        if len(selected) >= preset.max_peak_count:
            break
    return tuple(sorted(selected))


def _uneven_time_compress(
    audio: np.ndarray,
    sample_rate: int,
    protect_centers: tuple[int, ...],
    preset: EmotionPreset,
) -> np.ndarray:
    if not audio.size or not protect_centers:
        return _time_stretch(audio, sample_rate, preset.angry_weak_speed)
    half = int(sample_rate * preset.protect_window_ms / 1000 / 2)
    intervals = _merge_intervals((center - half, center + half) for center in protect_centers)
    parts: list[np.ndarray] = []
    cursor = 0
    for start, end in intervals:
        start = max(0, min(start, len(audio)))
        end = max(start, min(end, len(audio)))
        if start > cursor:
            parts.append(_time_stretch(audio[cursor:start], sample_rate, preset.angry_weak_speed))
        if end > start:
            parts.append(_time_stretch(audio[start:end], sample_rate, preset.angry_strong_speed))
        cursor = end
    if cursor < len(audio):
        parts.append(_time_stretch(audio[cursor:], sample_rate, preset.angry_weak_speed))
    return _concat_with_crossfades(parts, sample_rate, min(8.0, preset.crossfade_ms))


def _merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    prepared = sorted((int(start), int(end)) for start, end in intervals if int(end) > int(start))
    merged: list[tuple[int, int]] = []
    for start, end in prepared:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _local_positions(positions: tuple[int, ...], phrase_start: int, max_len: int) -> tuple[int, ...]:
    return tuple(position - phrase_start for position in positions if 0 <= position - phrase_start < max_len)


def _body_syllable_count(total_syllables: int, body_len: int, phrase_len: int) -> int:
    if total_syllables <= 0 or phrase_len <= 0:
        return 0
    return max(1, min(total_syllables, int(round(total_syllables * body_len / phrase_len))))


def _syllable_boundaries(length: int, syllable_count: int) -> list[tuple[int, int]]:
    if length <= 0:
        return []
    if syllable_count <= 1:
        return [(0, length)]
    edges = np.linspace(0, length, syllable_count + 1, dtype=int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(syllable_count) if edges[i + 1] > edges[i]]


def _syllable_time_stretch(
    audio: np.ndarray,
    sample_rate: int,
    syllable_count: int,
    rates: float | tuple[float, ...] | list[float],
    crossfade_ms: float,
) -> np.ndarray:
    if not audio.size:
        return audio
    if syllable_count <= 1:
        rate = float(np.mean(rates)) if not isinstance(rates, (float, int)) else float(rates)
        return _time_stretch(audio, sample_rate, rate)

    if isinstance(rates, (float, int)):
        rate_values = [float(rates)] * syllable_count
    else:
        rate_values = [float(rate) for rate in rates]
        if len(rate_values) < syllable_count:
            rate_values.extend([rate_values[-1] if rate_values else 1.0] * (syllable_count - len(rate_values)))
        rate_values = rate_values[:syllable_count]

    parts: list[np.ndarray] = []
    for (start, end), rate in zip(_syllable_boundaries(len(audio), syllable_count), rate_values, strict=False):
        segment = audio[start:end]
        if len(segment) < int(sample_rate * 0.05):
            parts.append(segment.copy())
        else:
            parts.append(_time_stretch(segment, sample_rate, rate))
    return _concat_with_crossfades(parts, sample_rate, min(crossfade_ms, 10.0))


def _angry_syllable_rates(syllable_count: int, preset: EmotionPreset) -> tuple[float, ...]:
    if syllable_count <= 0:
        return ()
    rates: list[float] = []
    for index in range(syllable_count):
        progress = index / max(1, syllable_count - 1)
        rate = preset.angry_weak_speed - 0.04 * progress
        if index >= syllable_count - 2:
            rate = preset.angry_strong_speed
        rates.append(float(rate))
    return tuple(rates)


def _angry_urgent_syllable_rates(syllable_count: int, preset: EmotionPreset) -> tuple[float, ...]:
    if syllable_count <= 0:
        return ()
    rates: list[float] = []
    for index in range(syllable_count):
        progress = index / max(1, syllable_count - 1)
        if progress < 0.42:
            rate = preset.angry_urgent_begin_speed
        elif progress < 0.78:
            local = (progress - 0.42) / 0.36
            rate = preset.angry_urgent_begin_speed + (preset.angry_urgent_middle_speed - preset.angry_urgent_begin_speed) * local
        else:
            local = (progress - 0.78) / 0.22
            rate = preset.angry_urgent_middle_speed + (preset.angry_strong_speed - preset.angry_urgent_middle_speed) * local
        rates.append(float(rate))
    return tuple(rates)


def _apply_syllable_gain_contour(
    audio: np.ndarray,
    syllable_count: int,
    start_db: float,
    mid_db: float,
    end_db: float,
) -> np.ndarray:
    if not audio.size or syllable_count <= 0:
        return audio
    centers = [(start + end) // 2 for start, end in _syllable_boundaries(len(audio), syllable_count)]
    if not centers:
        return audio
    x = np.array([0, *centers, len(audio) - 1], dtype=np.float32)
    center_progress = np.linspace(0.0, 1.0, len(centers), dtype=np.float32)
    center_db = np.interp(center_progress, [0.0, 0.62, 1.0], [start_db, mid_db, end_db])
    y = np.array([start_db * 0.35, *center_db, end_db * 0.5], dtype=np.float32)
    envelope_db = np.interp(np.arange(len(audio), dtype=np.float32), x, y)
    envelope_db = _smooth_1d(envelope_db, width=max(5, min(301, len(audio) // 12 or 5)))
    return audio * (10.0 ** (envelope_db / 20.0)).reshape(-1, 1)


def _smooth_phrase_onset(audio: np.ndarray, sample_rate: int, gain_db: float, duration_ms: float) -> np.ndarray:
    if not audio.size or abs(gain_db) < 0.01:
        return audio
    length = min(len(audio), max(1, int(sample_rate * duration_ms / 1000)))
    envelope_db = np.zeros(len(audio), dtype=np.float32)
    t = np.linspace(0.0, 1.0, length, dtype=np.float32)
    envelope_db[:length] = gain_db * (1.0 - t) ** 2
    return audio * (10.0 ** (envelope_db / 20.0)).reshape(-1, 1)


def _time_stretch(audio: np.ndarray, sample_rate: int, rate: float) -> np.ndarray:
    if not audio.size or abs(rate - 1.0) < 0.02:
        return audio.copy()
    target_len = max(1, int(round(len(audio) / rate)))
    if len(audio) < int(sample_rate * 0.08):
        return _fit_length(audio, target_len)
    return _ola_time_stretch(audio, sample_rate, rate, target_len)


def _ola_time_stretch(audio: np.ndarray, sample_rate: int, rate: float, target_len: int) -> np.ndarray:
    frame_len = int(sample_rate * 0.035)
    frame_len = max(256, min(frame_len, max(256, len(audio) // 2)))
    if frame_len % 2:
        frame_len += 1
    synth_hop = max(64, frame_len // 4)
    analysis_hop = max(1, int(round(synth_hop * rate)))
    window = np.hanning(frame_len).astype(np.float32).reshape(-1, 1)

    max_frames = max(1, int(np.ceil((len(audio) - frame_len) / max(1, analysis_hop))) + 2)
    out_len = max(target_len + frame_len * 2, synth_hop * max_frames + frame_len * 2)
    out = np.zeros((out_len, audio.shape[1]), dtype=np.float32)
    norm = np.zeros((out_len, 1), dtype=np.float32)

    in_pos = 0
    out_pos = 0
    while out_pos + frame_len <= out_len:
        if in_pos + frame_len <= len(audio):
            frame = audio[in_pos : in_pos + frame_len]
        else:
            frame = np.zeros((frame_len, audio.shape[1]), dtype=np.float32)
            available = max(0, len(audio) - in_pos)
            if available > 0:
                frame[:available] = audio[in_pos:]
        out[out_pos : out_pos + frame_len] += frame * window
        norm[out_pos : out_pos + frame_len] += window
        in_pos += analysis_hop
        out_pos += synth_hop
        if in_pos >= len(audio) and out_pos >= target_len:
            break

    valid_len = min(out_len, max(target_len, out_pos + frame_len))
    out = out[:valid_len]
    norm = norm[:valid_len]
    out = np.divide(out, np.maximum(norm, 1e-5), out=np.zeros_like(out), where=norm > 1e-5)
    return _fit_length(out, target_len)


def _pitch_shift(audio: np.ndarray, sample_rate: int, steps: float) -> np.ndarray:
    if not audio.size or abs(steps) < 0.1 or len(audio) < 2048:
        return audio.copy()
    channels = []
    for channel in range(audio.shape[1]):
        try:
            shifted = librosa.effects.pitch_shift(audio[:, channel].astype(np.float32), sr=sample_rate, n_steps=steps)
        except Exception:
            shifted = audio[:, channel].copy()
        channels.append(_fit_vector_length(shifted, len(audio)))
    return _stack_channels(channels)


def _falling_pitch_contour(audio: np.ndarray, sample_rate: int, fall_steps: float) -> np.ndarray:
    if len(audio) < 4096:
        return audio
    chunks = 4
    edges = np.linspace(0, len(audio), chunks + 1, dtype=int)
    out = audio.copy()
    for idx in range(chunks):
        start, end = int(edges[idx]), int(edges[idx + 1])
        if end - start < 1024:
            continue
        steps = -fall_steps * (idx / max(1, chunks - 1))
        out[start:end] = _pitch_shift(audio[start:end], sample_rate, steps)
    return out


def _apply_pitch_spikes(
    audio: np.ndarray,
    sample_rate: int,
    centers: tuple[int, ...],
    steps: float,
    width_ms: float,
) -> np.ndarray:
    if not audio.size or not centers:
        return audio
    out = audio.copy()
    half = int(sample_rate * width_ms / 1000 / 2)
    for center in sorted(set(centers)):
        start = max(0, center - half)
        end = min(len(out), center + half)
        if end - start < 2048:
            continue
        segment = out[start:end]
        shifted = _pitch_shift(segment, sample_rate, steps)
        window = np.hanning(end - start).reshape(-1, 1)
        out[start:end] = segment * (1.0 - window) + shifted * window
    return out


def _apply_gain_windows(
    audio: np.ndarray,
    sample_rate: int,
    centers: tuple[int, ...],
    gain_db: float,
    width_ms: float,
) -> np.ndarray:
    if not audio.size or not centers or abs(gain_db) < 0.01:
        return audio
    gain = 10.0 ** (gain_db / 20.0)
    envelope = np.ones(len(audio), dtype=np.float32)
    half = max(1, int(sample_rate * width_ms / 1000 / 2))
    for center in sorted(set(centers)):
        start = max(0, int(center) - half)
        end = min(len(audio), int(center) + half)
        if end <= start:
            continue
        window = np.hanning(end - start)
        envelope[start:end] = np.maximum(envelope[start:end], 1.0 + (gain - 1.0) * window)
    return audio * envelope.reshape(-1, 1)


def _tail_intensity_contour(audio: np.ndarray, start_gain_db: float, end_gain_db: float) -> np.ndarray:
    if not audio.size:
        return audio
    t = np.linspace(0.0, 1.0, len(audio), dtype=np.float32)
    rise_end = 0.18
    db = np.empty_like(t)
    rise = t <= rise_end
    if np.any(rise):
        db[rise] = np.interp(t[rise], [0.0, rise_end], [start_gain_db * 0.6, start_gain_db])
    if np.any(~rise):
        u = (t[~rise] - rise_end) / max(1e-6, 1.0 - rise_end)
        db[~rise] = start_gain_db + (end_gain_db - start_gain_db) * (u * u)
    return audio * (10.0 ** (db / 20.0)).reshape(-1, 1)


def _presence_boost(
    audio: np.ndarray,
    sample_rate: int,
    gain_db: float,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    if not audio.size or abs(gain_db) < 0.01 or len(audio) < 128:
        return audio
    nyquist = sample_rate / 2.0
    low = max(20.0, min(low_hz, nyquist * 0.8))
    high = max(low + 20.0, min(high_hz, nyquist * 0.95))
    if high >= nyquist or low >= high:
        return audio
    try:
        sos = butter(2, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
        band = sosfiltfilt(sos, audio, axis=0)
    except Exception:
        return audio
    delta = 10.0 ** (gain_db / 20.0) - 1.0
    return audio + band * delta


def _apply_gain_db(audio: np.ndarray, gain_db: float) -> np.ndarray:
    if not audio.size or abs(gain_db) < 0.01:
        return audio
    return audio * (10.0 ** (gain_db / 20.0))


def _loudness_normalize(audio: np.ndarray, target_dbfs: float, ceiling: float) -> np.ndarray:
    rms = float(np.sqrt(np.mean(audio * audio) + 1e-12))
    if rms <= 1e-7:
        return audio.copy()
    current_dbfs = 20.0 * np.log10(rms)
    gain_db = float(np.clip(target_dbfs - current_dbfs, -18.0, 12.0))
    normalized = _apply_gain_db(audio, gain_db)
    peak = float(np.max(np.abs(normalized))) if normalized.size else 0.0
    if peak > ceiling:
        normalized = normalized * (ceiling / peak)
    return normalized.astype(np.float32)


def _match_rms(audio: np.ndarray, reference: np.ndarray, max_gain_db: float = 6.0) -> np.ndarray:
    if not audio.size or not reference.size:
        return audio
    source_rms = float(np.sqrt(np.mean(audio * audio) + 1e-12))
    reference_rms = float(np.sqrt(np.mean(reference * reference) + 1e-12))
    if source_rms <= 1e-7 or reference_rms <= 1e-7:
        return audio
    gain_db = 20.0 * np.log10(reference_rms / source_rms)
    return _apply_gain_db(audio, float(np.clip(gain_db, -3.0, max_gain_db)))


def _soft_limiter(audio: np.ndarray, ceiling: float = 0.96) -> np.ndarray:
    if not audio.size:
        return audio
    limited = np.tanh(audio * 1.25) / np.tanh(1.25)
    peak = float(np.max(np.abs(limited)))
    if peak > ceiling:
        limited = limited * (ceiling / peak)
    return np.clip(limited, -ceiling, ceiling).astype(np.float32)


def _concat_with_crossfades(parts: list[np.ndarray], sample_rate: int, fade_ms: float) -> np.ndarray:
    valid = [part for part in parts if part.size]
    if not valid:
        return np.zeros((0, 1), dtype=np.float32)
    out = valid[0].astype(np.float32, copy=True)
    fade = int(sample_rate * fade_ms / 1000)
    for part in valid[1:]:
        part = part.astype(np.float32, copy=False)
        if not out.size:
            out = part.copy()
            continue
        n = min(fade, len(out), len(part))
        if n <= 1:
            out = np.vstack([out, part])
            continue
        left = np.linspace(1.0, 0.0, n, dtype=np.float32).reshape(-1, 1)
        right = 1.0 - left
        overlap = out[-n:] * left + part[:n] * right
        out = np.vstack([out[:-n], overlap, part[n:]])
    return out


def _fit_length(audio: np.ndarray, target_len: int) -> np.ndarray:
    target_len = max(0, int(target_len))
    if target_len == len(audio):
        return audio.copy()
    if target_len == 0:
        return np.zeros((0, audio.shape[1]), dtype=np.float32)
    if len(audio) <= 1:
        return np.zeros((target_len, audio.shape[1]), dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, len(audio), dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    out = np.empty((target_len, audio.shape[1]), dtype=np.float32)
    for channel in range(audio.shape[1]):
        out[:, channel] = np.interp(x_new, x_old, audio[:, channel]).astype(np.float32)
    return out


def _fit_vector_length(values: np.ndarray, target_len: int) -> np.ndarray:
    if len(values) == target_len:
        return values.astype(np.float32)
    return _fit_length(values.reshape(-1, 1), target_len)[:, 0]


def _stack_channels(channels: list[np.ndarray]) -> np.ndarray:
    max_len = max((len(channel) for channel in channels), default=0)
    if max_len == 0:
        return np.zeros((0, 1), dtype=np.float32)
    fixed = [_fit_vector_length(channel, max_len) for channel in channels]
    return np.stack(fixed, axis=1).astype(np.float32)


def _smooth_1d(values: np.ndarray, width: int) -> np.ndarray:
    if len(values) < width or width <= 1:
        return values.astype(np.float32)
    kernel = np.ones(width, dtype=np.float32) / width
    return np.convolve(values, kernel, mode="same").astype(np.float32)


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32)
    return np.mean(audio, axis=1).astype(np.float32)


def _read_wav_bytes(wav_bytes: bytes) -> tuple[np.ndarray, int, int]:
    with io.BytesIO(wav_bytes) as buffer:
        audio, sample_rate = sf.read(buffer, always_2d=True, dtype="float32")
    return audio.astype(np.float32), int(sample_rate), int(audio.shape[1])


def _write_wav_bytes(audio: np.ndarray, sample_rate: int, channels: int) -> bytes:
    output = io.BytesIO()
    data = audio[:, 0] if channels == 1 and audio.ndim == 2 else audio
    sf.write(output, data, sample_rate, format="WAV", subtype="PCM_16")
    return output.getvalue()
