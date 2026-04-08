from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional


TIMESTAMP_FORMAT = "%H-%M_%y%m%d"


def make_output_timestamp() -> str:
    return datetime.now().strftime(TIMESTAMP_FORMAT)


def make_timestamped_output_dir(parent: str, base_name: str) -> str:
    return os.path.join(parent, f"{base_name}_{make_output_timestamp()}")


def resolve_default_output_dir(parent: str, base_name: str, explicit: Optional[str] = None) -> str:
    if explicit:
        return os.path.abspath(explicit)
    return os.path.abspath(make_timestamped_output_dir(parent, base_name))


def _candidate_dirs(parent: str, base_name: str) -> list[Path]:
    parent_path = Path(parent)
    if not parent_path.is_dir():
        return []
    candidates: list[Path] = []
    exact = parent_path / base_name
    if exact.is_dir():
        candidates.append(exact)
    prefix = f"{base_name}_"
    for child in parent_path.iterdir():
        if child.is_dir() and child.name.startswith(prefix):
            candidates.append(child)
    return candidates


def find_latest_output_dir(parent: str, base_name: str) -> str:
    candidates = _candidate_dirs(parent, base_name)
    if not candidates:
        return os.path.abspath(os.path.join(parent, base_name))
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest.resolve())


def find_latest_output_file(parent: str, base_name: str, filename: str) -> str:
    latest_dir = find_latest_output_dir(parent, base_name)
    return os.path.abspath(os.path.join(latest_dir, filename))
