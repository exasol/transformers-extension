"""Nox tasks for starting the test-db and integration tests"""

import subprocess
import sys
from pathlib import Path

import nox
from exasol.toolbox.nox._shared import (
    get_filtered_python_files,
)

# imports all nox task provided by the toolbox
from exasol.toolbox.nox.tasks import *  # pylint: disable=wildcard-import disable=unused-wildcard-import
from nox import Session

from exasol_transformers_extension.deployment.language_container import (
    language_container_factory,
)
from exasol_transformers_extension.deployment.write_create_script import (
    write_create_script,
)
from noxconfig import (
    PROJECT_CONFIG,
)

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
        "test/integration_tests/with_db/deployment/test_create_script.py",
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


# These overridden functions should be removed as part of:
#    https://github.com/exasol/transformers-extension/issues/367


def _pyupgrade(session: Session, files: list[str]) -> None:
    session.run(
        "pyupgrade",
        "--py39-plus",
        "--exit-zero-even-if-changed",
        *files,
    )


def _code_format(session: Session, mode: Mode, files: list[str]) -> None:
    def command(*args: str) -> list[str]:
        return list(args) if mode == Mode.Fix else list(args) + ["--check"]

    session.run(*command("isort"), *files)
    session.run(*command("black"), *files)


@nox.session(name="format:fix", python=False)
def fix(session: Session) -> None:
    """Runs all automated fixes on the code base"""
    py_files = get_filtered_python_files(PROJECT_CONFIG.root_path)
    _pyupgrade(session, files=py_files)
    _code_format(session, Mode.Fix, py_files)
    write_create_script()


@nox.session(name="format:check", python=False)
def fmt_check(session: Session) -> None:
    """Checks the project for correct formatting"""
    py_files = get_filtered_python_files(PROJECT_CONFIG.root_path)
    _code_format(session=session, mode=Mode.Check, files=py_files)


def _git_create_script_up_to_date() -> int:
    """
    Check if "deployment/create_script.sql" needs to be changed and return the exit code of command git diff.
    The exit code is 0 if there are no changes.
    """
    p = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "-uno",
            "--",
            PROJECT_CONFIG.source_code_path / "deployment/create_script.sql",
        ],
        capture_output=True,
    )  # nosec: B603, B607 - fixed git command; PATH lookup and args are trusted here
    print(p.stdout.decode())
    return (
        False
        if "M exasol_transformers_extension/deployment/create_script.sql"
        in p.stdout.decode()
        else True
    )


@nox.session(name="create_script:updated", python=False)
def updated(_session: Session) -> None:
    """Checks if the create_script needs to be updated"""
    write_create_script()
    if not _git_create_script_up_to_date():
        print(
            "create_script changes when running write_create_script.\n"
            "Please run write_create_script and commit the resulting changes!"
            "(if you run 'nox -s format:fix' this gets fixed automatically)"
        )
        sys.exit(1)
