"""Configuration for nox based task runner"""

from __future__ import annotations
from __future__ import annotations

from pathlib import Path

from exasol.toolbox.config import BaseConfig
from pydantic import computed_field


class Config(BaseConfig):
    @computed_field  # type: ignore[misc]
    @property
    def source_code_path(self) -> Path:
        """
        Path to the source code of the project.
        This needs to be overridden due to a custom directory setup.
        """
        return self.root_path / self.project_name


PROJECT_CONFIG = Config(
    root_path=Path(__file__).parent,
    project_name="exasol_transformers_extension",
    python_versions=("3.10", "3.11", "3.12", "3.13", "3.14"),
    exasol_versions=("7.1.9",),
)
