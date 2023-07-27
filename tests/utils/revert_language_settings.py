import pyexasol
import pytest
from pytest_itde.config import TestConfig

from exasol_transformers_extension.deployment.language_container_deployer import \
    logger
import ssl


def revert_language_settings(func):
    def wrapper(**kwargs):
        try:
            return func(**kwargs)
        except Exception as exc:
            logger.debug("Exception occurred while running the test: %s" % exc)
            raise pytest.fail(exc)
        finally:
            logger.debug("Revert language settings")
            language_settings = kwargs['language_settings']
            itde:TestConfig = kwargs['itde']
            db_conn_revert = pyexasol.connect(
                dsn=f"{itde.db.host}:{itde.db.port}",
                user=itde.db.username,
                password=itde.db.password,
                encryption=True,
                websocket_sslopt={
                    "cert_reqs": ssl.CERT_NONE,
                }
            )
            db_conn_revert.execute(f"ALTER SYSTEM SET SCRIPT_LANGUAGES="
                                   f"'{language_settings[0][0]}';")
            db_conn_revert.execute(f"ALTER SESSION SET SCRIPT_LANGUAGES="
                                   f"'{language_settings[0][1]}';")

    return wrapper
