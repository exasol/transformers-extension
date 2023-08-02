import contextlib

from pyexasol import ExaConnection

from tests.utils.db_queries import DBQueries


@contextlib.contextmanager
def revert_language_settings(connection: ExaConnection):
    language_settings = DBQueries.get_language_settings(connection)
    try:
        yield
    finally:
        connection.execute(f"ALTER SYSTEM SET SCRIPT_LANGUAGES='{language_settings.system_value}';")
        connection.execute(f"ALTER SESSION SET SCRIPT_LANGUAGES='{language_settings.session_value}';")
