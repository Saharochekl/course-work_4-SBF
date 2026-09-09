"""Portable locations for code and saved products; no files are changed here.

Paths in old result JSON files may name the original workstation.  Interpret
those paths inside this checkout, never silently read a different old checkout.
Unrelated absolute paths (for example an external STPSF installation) stay
external.  Historical metadata and cache fingerprints are not rewritten on disk.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


# RU: корень от файла, а не cwd; EN: relocation must not depend on the shell's cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# RU: публичные команды и notebook. EN: package depth must not move entry points.
CODE_DIR = PROJECT_ROOT / "code"
# RU: единственный старый формат путей в сохранённых продуктах.
# EN: recognize the original checkout name, not arbitrary external directories.
_LEGACY_ROOT = "course_work-SBF"
# RU: только известные каталоги; EN: avoid interpreting ordinary text as a path.
_PROJECT_DIRS = {"code", "data", "runs", "texts", "materials", "output", "tmp"}


def project_path(value, root=None, *, must_exist=False) -> Path:
    """Resolve repository-relative or historical project paths, independent of CWD.

    ``root`` permits a renamed/relocated checkout and small isolated tests.
    ``must_exist`` is intended for input boundaries, not new output locations.
    """
    if value is None or str(value).strip() == "":
        raise ValueError("An empty value is not a project file path")
    root = PROJECT_ROOT if root is None else Path(root).expanduser().resolve()
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    elif not path.is_relative_to(root):
        anchors = [
            index for index, part in enumerate(path.parts[:-1])
            if part == _LEGACY_ROOT and path.parts[index + 1] in _PROJECT_DIRS
        ]
        if len(anchors) > 1:
            raise ValueError(f"Ambiguous historical project path: {value}")
        if anchors:
            path = root.joinpath(*path.parts[anchors[0] + 1:])
    path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Required input is absent: {path} (saved path: {value})")
    return path


def resolve_product_paths(payload, root=None):
    """Resolve only clearly identified file paths in an in-memory JSON structure.

    URLs, ordinary text, hashes, and unknown relative strings are left untouched.
    This does not alter serialized source records or manufacture missing files.
    """
    if isinstance(payload, dict):
        return {key: resolve_product_paths(value, root) for key, value in payload.items()}
    if isinstance(payload, list):
        return [resolve_product_paths(value, root) for value in payload]
    if isinstance(payload, str) and payload and "://" not in payload:
        path = Path(payload)
        if (
            len(path.parts) > 1 and path.parts[0] in _PROJECT_DIRS
            or path.is_absolute() and _LEGACY_ROOT in path.parts
        ):
            return str(project_path(payload, root))
    return payload


def load_project_json(path, root=None):
    """Read JSON with portable product locations without rewriting the source."""
    source = project_path(path, root, must_exist=True)
    return resolve_product_paths(json.loads(source.read_text(encoding="utf-8")), root)


def default_stpsf_data_dir() -> Path:
    """Prefer explicit STPSF_PATH, then local data; support existing home installs."""
    configured = os.environ.get("STPSF_PATH")
    if configured:
        return project_path(configured)
    local = PROJECT_ROOT / "data" / "stpsf-data"
    home_install = Path.home() / "data" / "stpsf-data"
    return home_install if not local.is_dir() and home_install.is_dir() else local
