import pyexasol
import pytest
from pytest_itde import config# todo not found where moved?


@pytest.fixture(scope="module")
def pyexasol_connection(connection_factory, exasol_config: config.Exasol) -> pyexasol.ExaConnection:
    connection = connection_factory(exasol_config)
    yield connection
    connection.close()
