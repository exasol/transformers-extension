import sys
from pathlib import Path
from exasol_transformers_extension.deployment.language_container import language_container_factory

sys.path += [str(Path().parent.absolute())]
ROOT_PATH = Path(__file__).parent
EXPORT_PATH = ROOT_PATH / "export"

import nox

"""defines nox tasks/targets for this project"""

# imports all nox task provided by the toolbox
from exasol.toolbox.nox.tasks import *  # pylint: disable=wildcard-import disable=unused-wildcard-import

# default actions to be run if nothing is explicitly specified with the -s option
nox.options.sessions = ["project:fix"]


@nox.session(python=False)
def export_slc(session: nox.Session):
    with language_container_factory() as container_builder:
        container_builder.export(EXPORT_PATH)

@nox.session(name="test:integration", python=False)
def TE_integration_test_overwrite(session) -> None:
    """Runs all integration tests with all backends"""
    # Overwrite for python toolbox tests:integration task, because we need additional parameters
    # todo find out if this could also be done with just sending parameters to test:integration task.
    # todo should this include starting db and export slc?
    # todo can we update itde to allow stopping with starting db seperateley?

    # We need to use an external database here, because the itde plugin doesn't provide all necessary options to
    # configure the database. See the start_database session.

    session.run('pytest', '--setup-show', '-s', '--backend=all', '--itde-db-version=external', 'test/integration_tests')


@nox.session(python=False)
def saas_integration_tests(session):
    # We need to use an external database here, because the itde plugin doesn't provide all necessary options to
    # configure the database. See the start_database session.
    session.run('pytest', '--setup-show', '-s', '--backend=saas', 'test/integration_tests/with_db')


@nox.session(python=False)
def onprem_integration_tests(session):
    # We need to use an external database here, because the itde plugin doesn't provide all necessary options to
    # configure the database. See the start_database session.
    session.run('pytest', '--setup-show', '-s', '--backend=onprem', '--itde-db-version=external', 'test/integration_tests/with_db')


@nox.session(python=False)
def without_db_integration_tests(session):
    # We need to use an external database here, because the itde plugin doesn't provide all necessary options to
    # configure the database. See the start_database session.
    session.run('pytest', '--setup-show', '-s', '--itde-db-version=external', 'test/integration_tests/without_db')


@nox.session(python=False)
def start_database(session):
    session.run('itde', 'spawn-test-environment',
                '--environment-name', 'test',
                '--database-port-forward', '8563',
                '--bucketfs-port-forward', '2580',
                '--db-mem-size', '8GB',
                '--nameserver', '8.8.8.8')
