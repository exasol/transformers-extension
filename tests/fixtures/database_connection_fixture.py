import pyexasol
import pytest
from pytest_itde.config import TestConfig


@pytest.fixture(scope="session")
def pyexasol_connection(itde: TestConfig) -> pyexasol.ExaConnection:
    return itde.ctrl_connection
