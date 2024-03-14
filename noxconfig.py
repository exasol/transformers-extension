"""Configuration for nox based task runner"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Iterable,
    MutableMapping,
)

from nox import Session, parametrize


@dataclass(frozen=True)
class Config:
    """Project specific configuration used by nox infrastructure"""

    root: Path = Path(__file__).parent
    doc: Path = Path(__file__).parent / "doc"
    version_file: Path = (
        Path(__file__).parent / "exasol_transformers_extension" / "version.py"
    )
    path_filters: Iterable[str] = ("dist", ".eggs", "venv", "metrics-schema")

    @staticmethod
    def pre_integration_tests_hook(
        _session: Session, _config: Config, _context: MutableMapping[str, Any]
    ) -> bool:
        # start the database
        _session.run(
            "itde",
            "spawn-test-environment",
            "--environment-name",
            "test",
            "--database-port-forward",
            "8888",
            "--bucketfs-port-forward",
            "6666",
            "--db-mem-size",
            "4GB",
            "--nameserver",
            "8.8.8.8",
        )
        #_config.
        _session.env.fromkeys({("--itde-db-version", "external")})
        _session.posargs.append("--itde-db-version")
        _session.posargs.append("external")
        _context.update({("--itde-db-version", "external")})
        parametrize(arg_values_list=["external"], arg_names=["--itde-db-version"])
        # todo probably run start_database here?

        return True

    @staticmethod
    def post_integration_tests_hook(
        _session: Session, _config: Config, _context: MutableMapping[str, Any]
    ) -> bool:
        """Implement if project specific behaviour is required"""
        return True


PROJECT_CONFIG = Config()
