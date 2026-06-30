"""Asset path helpers for MTCWorldMJX environments."""

from __future__ import annotations

from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = _PACKAGE_ROOT / "assets"


def asset_path(*parts: str) -> Path:
    """Return an absolute path under the package assets directory."""
    return ASSETS_DIR.joinpath(*parts)


def sawyer_xml_path(name: str) -> Path:
    """Return the path to a Sawyer XYZ task MJCF file."""
    if not name.endswith(".xml"):
        name = f"sawyer_{name}.xml" if not name.startswith("sawyer_") else f"{name}.xml"
    return asset_path("sawyer_xyz", name)
