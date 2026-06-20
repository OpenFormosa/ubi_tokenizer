"""Generic timestamped rich transcription formatting utilities."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any

TRANSCRIPT_START_TOKEN = "<|transcript_start|>"
TRANSCRIPT_END_TOKEN = "<|transcript_end|>"

NON_SPEECH_EVENT_ALIASES: dict[str, str] = {
    "silence": "[Silence]",
    "noise": "[Noise]",
    "music": "[Music]",
    "laughter": "[Laughter]",
    "breath": "[Breath]",
    "unintelligible_speech": "[Unintelligible Speech]",
    "environmental_sounds": "[Environmental Sounds]",
    "human_sounds": "[Human Sounds]",
}

SUPPORTED_NON_SPEECH_EVENTS: set[str] = set(NON_SPEECH_EVENT_ALIASES.values())


def format_timestamp_seconds(value: float, precision_digits: int = 2) -> str:
    """Format a non-negative finite timestamp as fixed-decimal seconds."""
    if precision_digits < 0:
        raise ValueError("precision_digits must be non-negative")
    if not math.isfinite(value):
        raise ValueError("timestamp must be finite")
    if value < 0:
        raise ValueError("timestamp must be non-negative")
    try:
        decimal_value = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("timestamp must be numeric") from exc
    quant = Decimal("1").scaleb(-precision_digits)
    rounded = decimal_value.quantize(quant, rounding=ROUND_HALF_UP)
    return f"{rounded:.{precision_digits}f}"


def format_transcript_segments_json(
    segments: list[dict],
    precision_digits: int = 2,
    compact: bool = True,
    ensure_ascii: bool = False,
) -> str:
    """Serialize transcript segments as JSON with numeric fixed-decimal timestamps."""
    normalized = [
        _normalize_segment(segment, precision_digits=precision_digits)
        for segment in segments
    ]
    if compact:
        serialized = [_serialize_segment(segment, ensure_ascii=ensure_ascii) for segment in normalized]
        return "[" + ",".join(serialized) + "]"
    serialized = [
        "  " + _serialize_segment(segment, ensure_ascii=ensure_ascii, pretty=True)
        for segment in normalized
    ]
    return "[\n" + ",\n".join(serialized) + "\n]"


def parse_transcript_segments_json(text: str) -> list[dict]:
    """Parse raw or transcript-delimited JSON segment text into a list of dicts."""
    payload = text.strip()
    if TRANSCRIPT_START_TOKEN in payload:
        start = payload.find(TRANSCRIPT_START_TOKEN) + len(TRANSCRIPT_START_TOKEN)
        end = payload.find(TRANSCRIPT_END_TOKEN, start)
        if end == -1:
            raise ValueError("Transcript start delimiter found without transcript end")
        payload = payload[start:end].strip()
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid transcript JSON: {exc.msg}") from exc
    if not isinstance(parsed, list):
        raise ValueError("Transcript JSON must be a list of segment objects")
    if not all(isinstance(item, dict) for item in parsed):
        raise ValueError("Transcript JSON segments must be objects")
    return parsed


def validate_transcript_segments(
    segments: list[dict],
    require_monotonic: bool = True,
    allow_overlap: bool = True,
    max_duration_seconds: float | None = None,
) -> dict:
    """Validate transcript segment shape and return errors, warnings, and summary."""
    errors: list[str] = []
    warnings: list[str] = []
    duration = 0.0
    if not isinstance(segments, list):
        return {
            "ok": False,
            "errors": ["segments must be a list"],
            "warnings": [],
            "num_segments": 0,
            "duration_seconds": 0.0,
        }

    previous_start: float | None = None
    previous_end: float | None = None
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            errors.append(f"segment {index} must be an object")
            continue
        start_value = _lookup(segment, "Start", "start")
        end_value = _lookup(segment, "End", "end")
        if start_value is None:
            errors.append(f"segment {index} is missing Start/start")
        if end_value is None:
            errors.append(f"segment {index} is missing End/end")
        start = _coerce_finite_number(start_value)
        end = _coerce_finite_number(end_value)
        if start is None:
            errors.append(f"segment {index} Start/start must be finite")
        if end is None:
            errors.append(f"segment {index} End/end must be finite")
        if start is not None and start < 0:
            errors.append(f"segment {index} Start/start must be >= 0")
        if start is not None and end is not None:
            if end < start:
                errors.append(f"segment {index} End/end must be >= Start/start")
            duration = max(duration, end)
            if require_monotonic and previous_start is not None and start < previous_start:
                errors.append(
                    f"segment {index} Start/start must be non-decreasing"
                )
            if not allow_overlap and previous_end is not None and start < previous_end:
                errors.append(f"segment {index} overlaps the previous segment")
            if max_duration_seconds is not None and end > max_duration_seconds:
                errors.append(
                    f"segment {index} End/end exceeds max_duration_seconds"
                )
            previous_start = start
            previous_end = end

        has_text = _lookup(segment, "Content", "text", "content") is not None
        event_value = _lookup(segment, "event")
        is_non_speech = event_value is not None or segment.get("Type") == "non_speech"
        if is_non_speech and event_value is not None:
            try:
                _canonical_non_speech_label(str(event_value))
            except ValueError as exc:
                errors.append(f"segment {index} {exc}")
        if not has_text and not is_non_speech:
            errors.append(f"segment {index} is missing Content/text")
        if "Start" not in segment and "start" not in segment:
            warnings.append(f"segment {index} formatted output should include Start")
        if "End" not in segment and "end" not in segment:
            warnings.append(f"segment {index} formatted output should include End")
        if not has_text and is_non_speech and event_value is None:
            warnings.append(
                f"segment {index} non-speech output should include Content"
            )

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "num_segments": len(segments),
        "duration_seconds": duration,
    }


def format_rich_transcription_prompt(
    instruction: str,
    input_audio_ref: str,
    context: list[str] | None = None,
    target_segments: list[dict] | None = None,
) -> str:
    """Generate a generic rich transcription prompt with optional target JSON."""
    lines = [
        f"<|system|>{instruction}",
        "<|task:speech_to_text|>",
        "<|user_channel|>",
        "<|input_audio_start|>",
        f"<|audio_ref_start|>{input_audio_ref}<|audio_ref_end|>",
        "<|input_audio_end|>",
    ]
    if context:
        context_objects = [{"type": "context", "text": item} for item in context]
        lines.extend(
            [
                "<|retrieval_result_start|>",
                json.dumps(context_objects, ensure_ascii=False, separators=(",", ":")),
                "<|retrieval_result_end|>",
            ]
        )
    lines.extend(["<|assistant_channel|>", TRANSCRIPT_START_TOKEN])
    if target_segments is not None:
        lines.append(format_transcript_segments_json(target_segments))
        lines.append(TRANSCRIPT_END_TOKEN)
    return "\n".join(lines)


def format_transcript_segments_structured(
    segments: list[dict],
    precision_digits: int = 2,
) -> str:
    """Serialize transcript segments with explicit structural delimiters."""
    normalized = [
        _normalize_segment(segment, precision_digits=precision_digits)
        for segment in segments
    ]
    lines = [TRANSCRIPT_START_TOKEN]
    for segment in normalized:
        content = str(segment["Content"])
        speaker = segment.get("Speaker")
        parts = [
            "<|segment_start|>",
            "<|speaker|>",
            "" if speaker is None else str(speaker),
            "<|start_time|>",
            str(segment["Start"]),
            "<|end_time|>",
            str(segment["End"]),
        ]
        if segment.get("Type") == "non_speech":
            parts.extend(["<|non_speech_event|>", content])
        else:
            parts.extend(["<|content|>", content])
        parts.append("<|segment_end|>")
        lines.append("".join(parts))
    lines.append(TRANSCRIPT_END_TOKEN)
    return "\n".join(lines)


def _normalize_segment(segment: Mapping[str, Any], precision_digits: int) -> dict[str, Any]:
    if not isinstance(segment, Mapping):
        raise ValueError("segment must be an object")
    start = _lookup(segment, "Start", "start")
    end = _lookup(segment, "End", "end")
    if start is None or end is None:
        raise ValueError("segment must include Start/start and End/end")
    speaker = _lookup(segment, "Speaker", "speaker")
    event = _lookup(segment, "event")
    if event is not None:
        content = _canonical_non_speech_label(str(event))
        return {
            "Start": format_timestamp_seconds(float(start), precision_digits),
            "End": format_timestamp_seconds(float(end), precision_digits),
            "Speaker": None,
            "Content": content,
            "Type": "non_speech",
        }
    content = _lookup(segment, "Content", "text", "content")
    if content is None:
        raise ValueError("speech segment must include Content/text")
    normalized: dict[str, Any] = {
        "Start": format_timestamp_seconds(float(start), precision_digits),
        "End": format_timestamp_seconds(float(end), precision_digits),
        "Speaker": speaker,
        "Content": str(content),
    }
    if segment.get("Type") == "non_speech":
        normalized["Type"] = "non_speech"
    return normalized


def _serialize_segment(
    segment: Mapping[str, Any],
    ensure_ascii: bool,
    pretty: bool = False,
) -> str:
    separators = (",", ": ") if pretty else (",", ":")
    item_separator, key_separator = separators
    pieces: list[str] = []
    for key in ("Start", "End", "Speaker", "Content", "Type"):
        if key not in segment:
            continue
        value = segment[key]
        if key in {"Start", "End"}:
            encoded_value = str(value)
        else:
            encoded_value = json.dumps(value, ensure_ascii=ensure_ascii)
        pieces.append(json.dumps(key) + key_separator + encoded_value)
    return "{" + item_separator.join(pieces) + "}"


def _canonical_non_speech_label(label: str) -> str:
    stripped = label.strip()
    if stripped in SUPPORTED_NON_SPEECH_EVENTS:
        return stripped
    key = stripped.lower().replace(" ", "_").replace("-", "_")
    if key in NON_SPEECH_EVENT_ALIASES:
        return NON_SPEECH_EVENT_ALIASES[key]
    raise ValueError(f"unsupported non-speech event label: {label}")


def _lookup(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _coerce_finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number
