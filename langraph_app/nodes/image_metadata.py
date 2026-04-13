"""
image_metadata_node — extract EXIF GPS/timestamp from image attachments.

Wraps legacy/get_img_metadata.py (read_image_metadata) which requires
a file path. We write each attachment to a temp file, call the legacy
function, then delete the temp file.

Runs in parallel with ocr_node and vision_node.
Only runs when has_attachments=True.

Conflict detection (flag-only, no automatic hoax penalty):
  - EXIF GPS is >5 miles from claimed location → location_mismatch
  - EXIF timestamp is >6 hours from time_start/time_end → timestamp_mismatch

Why flag-only: proxy reporters or old photos would trigger EXIF mismatch.
risk_assessment_node decides whether to act on conflicts based on other signals.
"""
from __future__ import annotations

import os
import tempfile
from typing import Optional

from legacy.get_img_metadata import read_image_metadata


_GPS_CONFLICT_MILES = 5.0
_TIME_CONFLICT_HOURS = 6.0


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in miles."""
    import math
    R = 3958.8  # Earth radius in miles
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def _extract_bytes(file_bytes: bytes, filename: str) -> dict:
    """Write bytes to a temp file, call read_image_metadata, return result."""
    suffix = os.path.splitext(filename)[-1] or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        return read_image_metadata(tmp_path)
    except Exception:
        return {}
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _check_conflicts(
    meta: dict,
    filename: str,
    claimed_location: list,
    time_start: str,
    time_end: str,
) -> list[dict]:
    conflicts: list[dict] = []

    gps = meta.get("gps", {})
    exif_lat = gps.get("latitude")
    exif_lon = gps.get("longitude")

    if exif_lat is not None and exif_lon is not None and len(claimed_location) == 2:
        claimed_lon, claimed_lat = claimed_location
        distance = _haversine_miles(exif_lat, exif_lon, claimed_lat, claimed_lon)
        if distance > _GPS_CONFLICT_MILES:
            conflicts.append({
                "type": "location_mismatch",
                "image_file": filename,
                "exif_value": f"{exif_lat:.5f},{exif_lon:.5f}",
                "claimed_value": f"{claimed_lat:.5f},{claimed_lon:.5f}",
                "delta": f"{distance:.1f} miles",
            })

    exif_data = meta.get("exif", {})
    exif_ts: Optional[str] = exif_data.get("DateTimeOriginal") or exif_data.get("DateTime")
    if exif_ts and (time_start or time_end):
        try:
            from datetime import datetime, timezone
            fmt = "%Y:%m:%d %H:%M:%S"
            exif_dt = datetime.strptime(exif_ts, fmt).replace(tzinfo=timezone.utc)
            ref_str = time_start or time_end
            ref_dt = datetime.fromisoformat(ref_str.replace("Z", "+00:00"))
            delta_hours = abs((exif_dt - ref_dt).total_seconds()) / 3600
            if delta_hours > _TIME_CONFLICT_HOURS:
                conflicts.append({
                    "type": "timestamp_mismatch",
                    "image_file": filename,
                    "exif_value": exif_ts,
                    "claimed_value": time_start or time_end,
                    "delta": f"{delta_hours:.1f} hours",
                })
        except (ValueError, TypeError):
            pass

    return conflicts


def run(state: dict) -> dict:
    if state.get("guardrail_hard_block"):
        return {}

    attachments: list = state.get("attachments", [])
    attachment_names: list = state.get("attachment_names", [])
    claimed_location: list = state.get("location", [])
    time_start: str = state.get("time_start", "")
    time_end: str = state.get("time_end", "")
    names = list(attachment_names) + ["unknown"] * len(attachments)

    all_metadata: list[dict] = []
    all_conflicts: list[dict] = []

    for idx, file_bytes in enumerate(attachments):
        if not isinstance(file_bytes, (bytes, bytearray)):
            all_metadata.append({})
            continue

        filename = names[idx]
        if filename.lower().endswith(".pdf"):
            all_metadata.append({})
            continue

        meta = _extract_bytes(bytes(file_bytes), filename)
        summary = {
            "file": filename,
            "gps_lat": meta.get("gps", {}).get("latitude"),
            "gps_lon": meta.get("gps", {}).get("longitude"),
            "timestamp": (meta.get("exif", {}).get("DateTimeOriginal")
                          or meta.get("exif", {}).get("DateTime")),
            "device": (
                f"{meta.get('exif', {}).get('Make', '')} "
                f"{meta.get('exif', {}).get('Model', '')}"
            ).strip() or None,
        }
        all_metadata.append(summary)

        conflicts = _check_conflicts(meta, filename, claimed_location, time_start, time_end)
        all_conflicts.extend(conflicts)

    return {
        "image_metadata": all_metadata,
        "image_metadata_conflicts": all_conflicts,
    }
