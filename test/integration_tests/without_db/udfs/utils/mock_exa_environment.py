from exasol_udf_mock_python.connection import Connection


class MockExaEnvironment:
    def __init__(self, connections: dict[str, Connection] = None):
        self._connections = connections
        if self._connections is None:
            self._connections = {}

    def get_connection(self, name: str) -> Connection:
        return self._connections[name]
