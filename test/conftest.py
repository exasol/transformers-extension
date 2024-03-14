"""defines fixtures to pytest-plugins"""
pytest_plugins = [
    "test.fixtures.database_connection_fixture",
    "test.fixtures.language_container_fixture",
    "test.fixtures.setup_database_fixture",
    "test.fixtures.bucketfs_fixture",
    "test.fixtures.model_fixture",
]
