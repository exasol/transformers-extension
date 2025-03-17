from __future__ import annotations

import logging
import pyexasol
from exasol.python_extension_common.connections.pyexasol_connection import open_pyexasol_connection

from exasol_transformers_extension.deployment.install_scripts_constants import InstallScriptsConstants
from exasol_transformers_extension.deployment.constants import constants
from exasol_transformers_extension.deployment.work_with_spans_constants import work_with_spans_constants
from exasol_transformers_extension.deployment.work_without_spans_constants import work_without_spans_constants
from exasol_transformers_extension.deployment import deployment_utils as utils

logger = logging.getLogger(__name__)


class ScriptsDeployer:
    def __init__(self, language_alias: str, schema: str,
                 pyexasol_conn: pyexasol.ExaConnection,
                 use_spans: bool = False, install_all_scripts: bool = False):
        self._language_alias = language_alias
        self._schema = schema
        self._use_spans = use_spans
        self._install_all_scripts = install_all_scripts
        self._pyexasol_conn = pyexasol_conn
        logger.debug("Init %s.", ScriptsDeployer.__name__)

    def _get_current_schema(self) -> str | None:
        return self._pyexasol_conn.execute("SELECT CURRENT_SCHEMA;").fetchval()

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
                "Could not create schema %s. Got error: %s", self._schema, e)
            logger.info("Trying to open schema %s instead.", self._schema)
        self._set_current_schema(self._schema)
        logger.info("Schema %s is opened.", self._schema)

    def  _deploy_udf_scripts_from_constants(self, constants_set: InstallScriptsConstants) -> None:
        for udf_call_src, template_src in constants_set.udf_callers_templates.items():
            udf_content = constants_set.udf_callers_dir.joinpath(
                udf_call_src).read_text()

            udf_query = utils.load_and_render_statement(
                template_src,
                script_content=udf_content,
                language_alias=self._language_alias,
                ordered_columns=constants_set.ordered_columns,
                work_with_spans=self._use_spans,
                install_all_scripts=self._install_all_scripts
            )

            self._pyexasol_conn.execute(udf_query)
            logger.debug("The UDF statement of the template %s is executed.", template_src)

    def _deploy_udf_scripts(self) -> None:
        """
        Deploy udf according to use_spans and install_all_scripts.
        Per default UDFs with and without spans are installed mutually exclusive.
        but setting install_all_scripts to true overrides this and installs all. This can be useful for testing.
        """
        install_scripts_constants = [constants]
        if self._use_spans or self._install_all_scripts:
            install_scripts_constants.append(work_with_spans_constants)
        if self._install_all_scripts or not self._use_spans:
            install_scripts_constants.append(work_without_spans_constants)
        for constant_set in install_scripts_constants:
            self._deploy_udf_scripts_from_constants(constant_set)


    def deploy_scripts(self) -> None:
        current_schema = self._get_current_schema()
        try:
            self._open_schema()
            self._deploy_udf_scripts()
            logger.debug("Scripts are deployed.")
        finally:
            self._set_current_schema(current_schema)

    @classmethod
    def run(cls,
            schema: str,
            language_alias: str,
            use_spans: bool = False,
            install_all_scripts: bool = False,
            **kwargs):

        pyexasol_conn = open_pyexasol_connection(**kwargs)
        scripts_deployer = cls(language_alias, schema,  pyexasol_conn,
                               use_spans, install_all_scripts)
        scripts_deployer.deploy_scripts()
