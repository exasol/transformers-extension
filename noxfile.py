import sys
from pathlib import Path
from exasol_transformers_extension.deployment.language_container import language_container_factory

sys.path += [str(Path().parent.absolute())]

import nox


ROOT_PATH = Path(__file__).parent
EXPORT_PATH = ROOT_PATH / "export"


@nox.session(python=False)
def export_slc(session: nox.Session):
    with language_container_factory() as container_builder:
        container_builder.export(EXPORT_PATH)


@nox.session(python=False)
def unit_tests(session):
    session.run('pytest', 'tests/unit_tests')


@nox.session(python=False)
def integration_tests(session):
    # We need to use a external database here, because the itde plugin doesn't provide all necessary options to
    # configure the database. See the start_database session.
    session.run('pytest', '--setup-show', '-s', '--backend=all', '--itde-db-version=external', 'tests/integration_tests')


@nox.session(python=False)
def saas_integration_tests(session):
    # We need to use a external database here, because the itde plugin doesn't provide all necessary options to
    # configure the database. See the start_database session.
    session.run('pytest', '--setup-show', '-s', '--backend=saas', 'tests/integration_tests/with_db')


@nox.session(python=False)
def onprem_integration_tests(session):
    # We need to use a external database here, because the itde plugin doesn't provide all necessary options to
    # configure the database. See the start_database session.
    session.run('pytest', '--setup-show', '-s', '--backend=onprem', '--itde-db-version=external', 'tests/integration_tests/with_db')


@nox.session(python=False)
def without_db_integration_tests(session):
    # We need to use a external database here, because the itde plugin doesn't provide all necessary options to
    # configure the database. See the start_database session.
    session.run('pytest', '--setup-show', '-s', '--itde-db-version=external', 'tests/integration_tests/without_db')


@nox.session(python=False)
def start_database(session):
    session.run('itde', 'spawn-test-environment',
                '--environment-name', 'test',
                '--database-port-forward', '8563',
                '--bucketfs-port-forward', '2580',
                '--db-mem-size', '8GB',
                '--nameserver', '8.8.8.8')
