"""
External package metadata detection.

Inspects uploaded files and extracts structured metadata heuristically.
Never executes files — read-only inspection of text content only.
"""
import json
from pathlib import Path
from typing import Any

ALLOWED_EXTENSIONS = {
    # Documents and data
    ".json", ".md", ".txt", ".csv",
    ".html", ".htm", ".pdf",
    # Images
    ".png", ".jpg", ".jpeg", ".gif", ".svg",
    # Audio artifacts (stored as-is, never executed)
    ".mp3", ".wav", ".flac", ".ogg", ".m4a",
    # Integrity and provenance
    ".sha256", ".sha512", ".md5",
}

TEXT_EXTENSIONS = {".json", ".md", ".txt", ".csv", ".html", ".htm"}

# Keys we look for in any JSON file
_RESEARCH_KEYS = {
    "experiment_id", "researcher", "author", "institution", "organization",
    "date", "methodology", "model", "judge_model", "description", "version",
    "contact", "license", "title", "abstract", "hypothesis",
}


def is_allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def mime_label(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    labels = {
        ".json": "JSON",
        ".md": "Markdown",
        ".txt": "Text",
        ".csv": "CSV",
        ".html": "HTML",
        ".htm": "HTML",
        ".pdf": "PDF",
        ".png": "Image",
        ".jpg": "Image",
        ".jpeg": "Image",
        ".gif": "Image",
        ".svg": "SVG",
        ".mp3": "Audio",
        ".wav": "Audio",
        ".flac": "Audio",
        ".ogg": "Audio",
        ".m4a": "Audio",
        ".sha256": "Checksum",
        ".sha512": "Checksum",
        ".md5": "Checksum",
    }
    return labels.get(ext, ext.lstrip(".").upper() or "Unknown")


def detect_metadata(files: list[dict]) -> dict[str, Any]:
    """
    files: list of {name, content_text (may be empty for binary)}
    Returns detected metadata dict.
    """
    meta: dict[str, Any] = {
        "file_count": len(files),
        "file_types": sorted({Path(f["name"]).suffix.lower() for f in files if f.get("name")}),
        "has_manifest": False,
        "has_readme": False,
    }

    for f in files:
        name_lower = Path(f.get("name", "")).name.lower()
        content = f.get("content_text", "") or ""
        if not content:
            continue

        if name_lower == "manifest.json":
            meta["has_manifest"] = True
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    meta["manifest"] = parsed
                    for key in _RESEARCH_KEYS:
                        if key in parsed and key not in meta:
                            meta[key] = parsed[key]
            except Exception:
                pass

        elif name_lower in ("readme.md", "readme.txt", "readme"):
            meta["has_readme"] = True
            lines = [l.strip() for l in content.split("\n") if l.strip()]
            if lines:
                title = lines[0].lstrip("#").strip()
                meta.setdefault("title", title)
            meta["readme_excerpt"] = "\n".join(lines[:15])

        elif name_lower.endswith(".json"):
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    for key in _RESEARCH_KEYS:
                        if key in parsed and key not in meta:
                            meta[key] = parsed[key]
            except Exception:
                pass

    return meta
