import contextlib

from pyexasol import ExaConnection

from tests.utils.db_queries import DBQueries


@contextlib.contextmanager
def revert_language_settings(connection: ExaConnection):
    language_settings = DBQueries.get_language_settings(connection)
    try:
        yield
    finally:
        connection.execute(f"ALTER SYSTEM SET SCRIPT_LANGUAGES='{language_settings[0][0]}';")
        connection.execute(f"ALTER SESSION SET SCRIPT_LANGUAGES='{language_settings[0][1]}';")
