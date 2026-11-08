"""Windows long-path helpers for DICOM studies with deeply nested folders."""

from __future__ import annotations

import os
from pathlib import Path


def long_path(path: str | os.PathLike[str]) -> Path:
    """Return a path that bypasses the traditional Windows MAX_PATH limit.

    On Windows this uses the extended-length ``\\\\?\\`` namespace. On other
    platforms the path is unchanged. Keep the returned path for filesystem
    traversal and DICOM reads; do not strip the prefix before opening files.
    """
    raw = os.path.expanduser(os.fspath(path))
    if os.name != "nt":
        return Path(raw).resolve()
    if raw.startswith("\\\\?\\"):
        return Path(raw)

    if os.path.isabs(raw):
        absolute = os.path.normpath(raw)
    else:
        # Build an absolute string without asking Windows to stat/resolve the
        # potentially over-MAX_PATH relative input first.
        absolute = os.path.normpath(os.path.join(os.getcwd(), raw))
    if absolute.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + absolute.lstrip("\\"))
    return Path("\\\\?\\" + absolute)
