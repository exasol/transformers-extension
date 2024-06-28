from __future__ import annotations

import logging
import pyexasol
from exasol.python_extension_common.connections.pyexasol_connection import open_pyexasol_connection

from exasol_transformers_extension.deployment import constants, \
    deployment_utils as utils

logger = logging.getLogger(__name__)


class ScriptsDeployer:
    def __init__(self, language_alias: str, schema: str,
                 pyexasol_conn: pyexasol.ExaConnection):
        self._language_alias = language_alias
        self._schema = schema
        self._pyexasol_conn = pyexasol_conn
        logger.debug(f"Init {ScriptsDeployer.__name__}.")

    def _get_current_schema(self) -> str | None:
        return self._pyexasol_conn.execute(f"SELECT CURRENT_SCHEMA;").fetchval()

    def _set_current_schema(self, schema: str | None):
        if schema:
            self._pyexasol_conn.execute(f'OPEN SCHEMA "{schema}";')
        else:
            self._pyexasol_conn.execute("CLOSE SCHEMA;")

    def _open_schema(self) -> None:
        try:
            self._pyexasol_conn.execute(
                f'CREATE SCHEMA IF NOT EXISTS "{self._schema}";')
        except pyexasol.ExaQueryError as e:
            logger.warning(
                f"Could not create schema {self._schema}. Got error: {e}")
            logger.info(f"Trying to open schema {self._schema} instead.")
        self._set_current_schema(self._schema)
        logger.info(f"Schema {self._schema} is opened.")

    def _deploy_udf_scripts(self) -> None:
        for udf_call_src, template_src in constants.UDF_CALL_TEMPLATES.items():
            udf_content = constants.UDF_CALLERS_DIR.joinpath(
                udf_call_src).read_text()
            udf_query = utils.load_and_render_statement(
                template_src,
                script_content=udf_content,
                language_alias=self._language_alias,
                ordered_columns=constants.ORDERED_COLUMNS)

            self._pyexasol_conn.execute(udf_query)
            logger.debug(f"The UDF statement of the template "
                         f"{template_src} is executed.")

    def deploy_scripts(self) -> None:
        current_schema = self._get_current_schema()
        try:
            self._open_schema()
            self._deploy_udf_scripts()
            logger.debug(f"Scripts are deployed.")
        finally:
            self._set_current_schema(current_schema)

    @classmethod
    def run(cls,
            schema: str,
            language_alias: str,
            **kwargs):

        pyexasol_conn = open_pyexasol_connection(**kwargs)
        scripts_deployer = cls(language_alias, schema, pyexasol_conn)
        scripts_deployer.deploy_scripts()
