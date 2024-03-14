import sys
from pathlib import Path

import nox

# imports all nox task provided by the python_toolbox
from exasol.toolbox.nox.tasks import *  # pylint: disable=wildcard-import disable=unused-wildcard-import
from exasol.toolbox.nox.tasks import type_check

# default actions to be run if nothing is explicitly specified with the -s option
nox.options.sessions = ["fix"]

sys.path += [str(Path().parent.absolute())]

from exasol_transformers_extension.deployment.language_container import (
    build_language_container,
    export,
    find_flavor_path,
    prepare_flavor,
)

ROOT_PATH = Path(__file__).parent
EXPORT_PATH = ROOT_PATH / "export"


@nox.session(python=False)
def build_slc(session: nox.Session):
    flavor_path = find_flavor_path()
    prepare_flavor(flavor_path)
    build_language_container(flavor_path)


@nox.session(python=False)
def export_slc(session: nox.Session):
    flavor_path = find_flavor_path()
    prepare_flavor(flavor_path)
    export(flavor_path, EXPORT_PATH)

@nox.session(python=False)
def integration_tests(session):
    # We need to use a external database here, because the itde plugin doesn't provide
    # all necassary options to
    # configure the database. See the start_database session.
    session.run("pytest", "--itde-db-version=external", "test/integration")
