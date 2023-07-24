import pytest
import pyexasol
from tests.utils.parameters import db_params
import ssl


@pytest.fixture(scope="session")
def pyexasol_connection() -> pyexasol.ExaConnection:
    conn = pyexasol.connect(
        dsn=db_params.address(),
        user=db_params.user,
        password=db_params.password,
        encryption=True,
        websocket_sslopt={
            "cert_reqs": ssl.CERT_NONE,
        }
    )
    return conn
