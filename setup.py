#!/usr/bin/env python3
"""Setuptools entry point for MTCWorldMJX.

Metadata and dependencies are defined in ``pyproject.toml``. This file ensures
MuJoCo MJCF/mesh assets under ``MTCWorldMJX/assets/`` are included in
wheels and editable installs.
"""

from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup

_ROOT = Path(__file__).resolve().parent
_ASSETS_ROOT = _ROOT / "MTCWorldMJX" / "assets"


def _package_asset_files() -> list[str]:
    """Paths relative to the ``MTCWorldMJX`` package directory."""
    if not _ASSETS_ROOT.is_dir():
        return []
    pkg_root = _ASSETS_ROOT.parent
    return [
        str(path.relative_to(pkg_root))
        for path in _ASSETS_ROOT.rglob("*")
        if path.is_file()
    ]


setup(
    name="MTCWorldMJX",
    packages=find_packages(where=".", include=["MTCWorldMJX*"]),
    package_dir={"": "."},
    package_data={"MTCWorldMJX": _package_asset_files()},
    include_package_data=True,
)
