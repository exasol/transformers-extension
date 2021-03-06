import pyexasol
from exasol_transformers_extension.deployment import constants, utils
import logging
logger = logging.getLogger(__name__)


class ScriptsDeployer:
    def __init__(self, language_alias: str, schema: str,
                 pyexasol_conn: pyexasol.ExaConnection):
        self._language_alias = language_alias
        self._schema = schema
        self._pyexasol_conn = pyexasol_conn
        logger.debug(f"Init {ScriptsDeployer.__name__}.")

    def _open_schema(self) -> None:
        queries = ["CREATE SCHEMA IF NOT EXISTS {schema_name}",
                   "OPEN SCHEMA {schema_name}"]
        for query in queries:
            self._pyexasol_conn.execute(query.format(schema_name=self._schema))
        logger.debug(f"Schema {self._schema} is opened.")

    def _deploy_udf_scripts(self) -> None:
        for udf_call_src, template_src in constants.UDF_CALL_TEMPLATES.items():
            udf_content = constants.SOURCE_DIR.joinpath(
                udf_call_src).read_text()
            udf_query = utils.load_and_render_statement(
                template_src,
                script_content=udf_content,
                language_alias=self._language_alias)
            self._pyexasol_conn.execute(udf_query)
            logger.debug(f"The UDF statement of the template "
                         f"{template_src} is executed.")

    def deploy_scripts(self) -> None:
        self._open_schema()
        self._deploy_udf_scripts()
        logger.debug(f"Scripts are deployed.")

    @classmethod
    def run(cls, dsn: str, user: str, password: str,
            schema: str, language_alias: str):

        pyexasol_conn = pyexasol.connect(dsn=dsn, user=user, password=password)
        scripts_deployer = cls(language_alias, schema, pyexasol_conn)
        scripts_deployer.deploy_scripts()
