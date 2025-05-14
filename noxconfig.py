"""Configuration for nox based task runner"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Project specific configuration used by nox infrastructure"""

    root: Path = Path(__file__).parent
    doc: Path = Path(__file__).parent / "doc"
    importlinter: Path = Path(__file__).parent / ".import_linter_config"
    version_file: Path = Path(__file__).parent / "version.py"
    path_filters: Iterable[str] = (
        "dist",
        ".eggs",
        "venv",
        "metrics-schema",
        "idioms",
    )
    python_versions = ["3.10", "3.11", "3.12", "3.13"]
    exasol_versions = ["7.1.9"]
    plugins: Iterable = ()  # [UpdateTemplates]


PROJECT_CONFIG = Config()
