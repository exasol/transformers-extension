pytest_plugins = [
    "tests.fixtures.database_connection_fixture",
    "tests.fixtures.script_deployment_fixture",
    "tests.fixtures.bucketfs_fixture",
    "tests.fixtures.language_container_fixture",
    "tests.fixtures.setup_database_fixture",
    "tests.fixtures.model_fixture",
]

_BACKEND_OPTION = '--backend'


def pytest_addoption(parser):
    parser.addoption(
        _BACKEND_OPTION,
        action="append",
        default=[],
        help=f"""List of test backends (onprem, saas). By default, the tests will be
            run on both backends. To select only one of the backends add the
            argument {_BACKEND_OPTION}=<name-of-the-backend> to the command line. Both
            backends can be selected like ... {_BACKEND_OPTION}=onprem {_BACKEND_OPTION}=saas,
            but this is the same as the default.
            """,
    )
