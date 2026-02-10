"""Nox tasks for starting the test-db and integration tests"""

import sys
from pathlib import Path

import nox
from exasol.toolbox.nox._shared import (
    _version,
    get_filtered_python_files,
)

# imports all nox task provided by the toolbox
from exasol.toolbox.nox.tasks import *  # pylint: disable=wildcard-import disable=unused-wildcard-import
from nox import Session

from exasol_transformers_extension.deployment.language_container import (
    language_container_factory,
)
from noxconfig import PROJECT_CONFIG

sys.path += [str(Path().parent.absolute())]
ROOT_PATH = Path(__file__).parent
EXPORT_PATH = ROOT_PATH / "export"

# default actions to be run if nothing is explicitly specified with the -s option
nox.options.sessions = ["format:fix"]


@nox.session(python=False)
def export_slc(session: nox.Session):
    """Exports Transformers Extension Script Language Container"""
    with language_container_factory() as container_builder:
        container_builder.export(EXPORT_PATH)


@nox.session(name="test:integration", python=False)
def te_integration_test_overwrite(session) -> None:
    """Runs all integration tests with all backends"""
    # Overwrite for python toolbox tests:integration task,
    # because we need additional parameters

    # We need to use an external database here, because the itde plugin doesn't
    # provide all necessary options to
    # configure the database. See the start_database session.

    session.run(
        "pytest",
        "--setup-show",
        "-s",
        "--backend=all",
        "--itde-db-version=external",
        "test/integration_tests",
    )


@nox.session(python=False)
def saas_integration_tests(session):
    """
    Runs all integration tests with saas backend
    """
    # We need to use an external database here, because the itde plugin doesn't
    # provide all necessary options to
    # configure the database. See the start_database session.
    session.run(
        "pytest",
        "--setup-show",
        "-s",
        "--backend=saas",
        "test/integration_tests/with_db",
    )


@nox.session(python=False)
def onprem_integration_tests(session):
    """
    Runs all integration tests with onprem backend
    """
    # We need to use an external database here, because the itde plugin doesn't
    # provide all necessary options to
    # configure the database. See the start_database session.
    session.run(
        "pytest",
        "--setup-show",
        "-s",
        "--backend=onprem",
        "--itde-db-version=external",
        "test/integration_tests/with_db",
    )


@nox.session(python=False)
def without_db_integration_tests(session):
    """
    Runs only non-db integration tests
    """
    # We need to use an external database here, because the itde plugin doesn't
    # provide all necessary options to
    # configure the database. See the start_database session.
    session.run(
        "pytest",
        "--setup-show",
        "-s",
        "--itde-db-version=external",
        "test/integration_tests/without_db",
    )


@nox.session(python=False)
def start_database(session):
    """
    Starts onprem backend/db
    """
    session.run(
        "itde",
        "spawn-test-environment",
        "--environment-name",
        "test",
        "--database-port-forward",
        "8563",
        "--bucketfs-port-forward",
        "2580",
        "--db-mem-size",
        "8GB",
        "--nameserver",
        "8.8.8.8",
    )


def _pyupgrade(session: Session, files: list[str]) -> None:
    session.run(
        "pyupgrade",
        "--py39-plus",
        "--exit-zero-even-if-changed",
        *files,
    )


def _code_format(session: Session, mode: Mode, files: list[str]) -> None:
    def command(*args: str) -> list[str]:
        return args if mode == Mode.Fix else list(args) + ["--check"]

    session.run(*command("isort"), *files)
    session.run(*command("black"), *files)


@nox.session(name="format:fix", python=False)
def fix(session: Session) -> None:
    """Runs all automated fixes on the code base"""
    py_files = get_filtered_python_files(PROJECT_CONFIG.root_path)
    _version(session, Mode.Fix)
    _pyupgrade(session, files=py_files)
    _code_format(session, Mode.Fix, py_files)


@nox.session(name="format:check", python=False)
def fmt_check(session: Session) -> None:
    """Checks the project for correct formatting"""
    py_files = get_filtered_python_files(PROJECT_CONFIG.root_path)
    _code_format(session=session, mode=Mode.Check, files=py_files)
