from exasol_udf_mock_python.connection import Connection

# this is the same as exasol_udf_mock_python.mock_exa_environment
# if it accepted metadata=None as input #todo change the exasol_udf_mock_python?
class MockExaEnvironment:
    def __init__(self, connections: dict[str, Connection] = None):
        self._connections = connections
        if self._connections is None:
            self._connections = {}

    def get_connection(self, name: str) -> Connection:
        return self._connections[name]