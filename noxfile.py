import nox


@nox.session(python=False)
def unit_tests(session):
    session.run('pytest', 'tests/unit_tests')


@nox.session(python=False)
def integration_tests(session):
    # We need to use a external database here, because the itde plugin doesn't provide all necassary options to
    # configure the database. See the start_database session.
    session.run('pytest', '--itde-db-version=external', 'tests/integration_tests')


@nox.session(python=False)
def start_database(session):
    session.run('itde', 'spawn-test-environment',
                '--environment-name', 'test',
                '--database-port-forward', '8888',
                '--bucketfs-port-forward', '6666',
                '--db-mem-size', '4GB',
                '--nameserver', '8.8.8.8')
