# pip install Pillow
# Recommended for Apple HEIC/HEIF: pip install pillow-heif
from PIL import Image, ExifTags, ImageOps
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Union

# --- Optional: enable HEIC/HEIF support if the library is installed ---
def _try_register_heif():
    try:
        from pillow_heif import register_heif_opener  # type: ignore
        register_heif_opener()
    except Exception:
        # pillow-heif not installed or failed to load; HEIC may not open
        pass

_try_register_heif()

# Build a reverse EXIF tag map once
_EXIF_TAGS = {v: k for k, v in ExifTags.TAGS.items()}

def _to_deg(value) -> float:
    """Convert EXIF GPS coordinates (DMS rational tuples) to decimal degrees."""
    d = float(value[0][0]) / float(value[0][1])
    m = float(value[1][0]) / float(value[1][1])
    s = float(value[2][0]) / float(value[2][1])
    return d + (m / 60.0) + (s / 3600.0)

def _parse_gps(exif) -> Optional[Dict[str, float]]:
    """Extract decimal latitude/longitude from EXIF GPS IFD if present."""
    gps_tag_id = _EXIF_TAGS.get("GPSInfo")
    if gps_tag_id is None:
        return None
    gps_ifd = exif.get(gps_tag_id)
    if not gps_ifd:
        return None

    # GPS keys in sub-IFD use numeric codes; normalize to names
    gps_tags: Dict[str, Any] = {}
    for k, v in gps_ifd.items():
        tag_name = ExifTags.GPSTAGS.get(k, k)
        gps_tags[tag_name] = v

    try:
        lat = _to_deg(gps_tags["GPSLatitude"])
        if gps_tags.get("GPSLatitudeRef") in ("S", b"S"):
            lat = -lat
        lon = _to_deg(gps_tags["GPSLongitude"])
        if gps_tags.get("GPSLongitudeRef") in ("W", b"W"):
            lon = -lon
        return {"latitude": lat, "longitude": lon}
    except Exception:
        return None

def _readable_exif_value(value: Any) -> Union[str, float, int]:
    """
    Normalize EXIF values to JSON-friendly primitives:
      - bytes -> decoded UTF-8 (ignore errors)
      - rationals -> float(value)
      - tuples/lists of rationals -> readable string
      - else -> primitive or repr(...)
    """
    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode("utf-8", errors="ignore")
        except Exception:
            return repr(value)

    # Handle PIL IFDRational
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        try:
            return float(value)  # type: ignore[arg-type]
        except Exception:
            pass

    if isinstance(value, (tuple, list)):
        try:
            def _maybe_float(x):
                if hasattr(x, "numerator") and hasattr(x, "denominator"):
                    return float(x)
                if (
                    isinstance(x, (tuple, list))
                    and len(x) == 2
                    and all(isinstance(y, (int, float)) for y in x)
                ):
                    num, den = x
                    return float(num) / float(den) if den else float(num)
                return x
            converted = [_maybe_float(x) for x in value]
            return str(converted)
        except Exception:
            return repr(value)

    if isinstance(value, (str, int, float)):
        return value

    return repr(value)

def _detect_live_photo_pair(image_path: Path) -> Dict[str, Optional[str]]:
    """
    Detect Apple Live Photo pairing by checking for a sibling video (.mov/.mp4)
    with the same stem. Returns {"is_live_photo": bool, "paired_video": str|None}.
    """
    stem = image_path.stem
    parent = image_path.parent
    candidates = [
        parent / f"{stem}.mov",
        parent / f"{stem}.MOV",
        parent / f"{stem}.mp4",
        parent / f"{stem}.MP4",
    ]
    for c in candidates:
        if c.exists():
            return {"is_live_photo": True, "paired_video": str(c)}
    return {"is_live_photo": False, "paired_video": None}

def _apple_extras(make: Optional[str], model: Optional[str], image_path: Path) -> Dict[str, Any]:
    """
    Add Apple-specific hints:
      - detect Live Photo pairing
      - flag likely Apple source from Make/Model or extension
    """
    lower_ext = image_path.suffix.lower()
    is_heic = lower_ext in {".heic", ".heif"}
    is_dng = lower_ext == ".dng"

    looks_apple = bool(
        (make and isinstance(make, str) and make.strip().lower() == "apple")
        or (model and isinstance(model, str) and "iphone" in model.lower())
        or is_heic
        or is_dng
    )

    live = _detect_live_photo_pair(image_path) if is_heic else {"is_live_photo": False, "paired_video": None}

    return {
        "looks_apple": looks_apple,
        "is_heic": is_heic,
        "is_dng_proraw": is_dng,
        **live,
    }

def read_image_metadata(path: str) -> Dict[str, Any]:
    """
    Returns a dict with:
      - file: name, size_bytes, modified_time
      - image: format, width, height, mode (after EXIF orientation fix)
      - exif: flattened EXIF key→value dict (strings/numbers only)
      - gps: {latitude, longitude} if available
      - png_info: metadata keys present for PNG (e.g., dpi)
      - apple: Apple-specific hints (HEIC/ProRAW detection, Live Photo pairing)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    # File system basics
    stat = p.stat()
    file_info = {
        "name": p.name,
        "size_bytes": stat.st_size,
        "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "path": str(p),
    }

    with Image.open(p) as im:
        # Keep original format, but use EXIF transpose for correct dimensions
        fmt = im.format
        try:
            im_view = ImageOps.exif_transpose(im)
        except Exception:
            im_view = im  # fallback

        image_info = {
            "format": fmt,
            "width": im_view.width,
            "height": im_view.height,
            "mode": im_view.mode,
        }

        # EXIF
        exif_data: Dict[str, Any] = {}
        raw_exif = im.getexif() or {}

        for tag_id, value in raw_exif.items():
            tag = ExifTags.TAGS.get(tag_id, str(tag_id))
            exif_data[tag] = _readable_exif_value(value)

        # GPS
        gps = _parse_gps(raw_exif)

        # PNG ancillary info
        png_info: Optional[Dict[str, Any]] = None
        if fmt == "PNG" and hasattr(im, "info") and im.info:
            png_info = {}
            for k, v in im.info.items():
                if k == "icc_profile" and isinstance(v, (bytes, bytearray)):
                    png_info["icc_profile"] = f"<{len(v)} bytes>"
                else:
                    png_info[k] = v

        # Apple-specific extras (HEIC/Live Photo/ProRAW)
        apple = _apple_extras(
            make=exif_data.get("Make"),
            model=exif_data.get("Model"),
            image_path=p,
        )

    return {
        "file": file_info,
        "image": image_info,
        "exif": exif_data,
        "gps": gps,
        "png_info": png_info,
        "apple": apple,
    }

# --- Example usage ---
if __name__ == "__main__":
    # Hardcode a single image file name that lives next to this script in ./images/
    HERE = Path(__file__).resolve().parent
    IMG_DIR = HERE / "images"
    PATH = IMG_DIR / "IMG_0270.HEIC"   # <- change this (supports .jpg/.png/.heic/.dng, etc.)

    meta = read_image_metadata(str(PATH))
    from pprint import pprint
    pprint(meta)
