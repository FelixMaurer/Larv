from __future__ import annotations

import hashlib
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Iterable

from .config import Config
from .qr import QRResult


def collect_images(path: Path, cfg: Config, recursive: bool = False) -> list[Path]:
    """Return sorted image files from a single file or directory."""
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() not in cfg.image_extensions:
            raise ValueError(f"Unsupported image extension: {path.suffix}")
        return [path]

    iterator: Iterable[Path] = path.rglob("*") if recursive else path.iterdir()
    images = [p for p in iterator if p.is_file() and p.suffix.lower() in cfg.image_extensions]
    return sorted(images)


def safe_slug(text: str, max_len: int = 80) -> str:
    """Make a filesystem-safe, human-readable slug."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.strip().strip("_")
    text = re.sub(r"[^A-Za-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    if not text:
        return "NA"
    return text[:max_len].rstrip("-")


def short_hash(text: str, length: int = 8) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:length]


def output_basename(image_path: Path, qr: QRResult) -> str:
    """Basename used for all derived outputs.

    The original filename is preserved, and the decoded QR text is embedded as
    a slug. A short hash prevents collisions when many files share the same QR.
    """
    original = safe_slug(image_path.stem, max_len=60)
    if qr.detected and qr.text:
        qr_slug = safe_slug(qr.text, max_len=80)
    else:
        qr_slug = "NOQR"
    return f"{original}__qr-{qr_slug}__{short_hash(str(image_path.resolve()))}"


def copy_original_with_metadata_name(image_path: Path, destination_dir: Path, basename: str) -> Path:
    destination = destination_dir / f"{basename}{image_path.suffix.lower()}"
    shutil.copy2(image_path, destination)
    return destination
