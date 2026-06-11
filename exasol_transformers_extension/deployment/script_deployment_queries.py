"""Class for reading constants files, and converting them to SQL queries for the installation of the UDF's"""

from exasol_transformers_extension.deployment import deployment_utils as utils
from exasol_transformers_extension.deployment.constants import constants
from exasol_transformers_extension.deployment.install_scripts_constants import (
    InstallScriptsConstants,
)
from exasol_transformers_extension.deployment.work_with_spans_constants import (
    work_with_spans_constants,
)
from exasol_transformers_extension.deployment.work_without_spans_constants import (
    work_without_spans_constants,
)


class ScriptDeploymentQueries:
    """Class for reading constants files, and converting them to SQL queries for the installation of the UDF's"""

    def __init__(
        self,
        language_alias: str,
        use_spans: bool = False,
        install_all_scripts: bool = False,
    ):
        self._language_alias = language_alias
        self._use_spans = use_spans
        self._install_all_scripts = install_all_scripts

    def get_constant_set(self) -> list[InstallScriptsConstants]:
        """Returns a list of constants defined in the constants file.
        "_install_all_scripts" and "_use_spans" can be set to define which constants are returned.
        """
        install_scripts_constants = [constants]
        if self._use_spans or self._install_all_scripts:
            install_scripts_constants.append(work_with_spans_constants)
        if self._install_all_scripts or not self._use_spans:
            install_scripts_constants.append(work_without_spans_constants)
        return install_scripts_constants

    def make_script_deployment_queries(self, constants_set: InstallScriptsConstants):
        """
        Creates SQL-queries from the given constants set.
        These queries can be called to install the udfs.
        """
        queries = {}
        for udf_call_src, template_src in constants_set.udf_callers_templates.items():
            udf_content = constants_set.udf_callers_dir.joinpath(
                udf_call_src
            ).read_text()

            udf_query = utils.load_and_render_statement(
                template_src,
                script_content=udf_content,
                language_alias=self._language_alias,
                ordered_columns=constants_set.ordered_columns,
                work_with_spans=self._use_spans,
                install_all_scripts=self._install_all_scripts,
            )
            queries[template_src] = udf_query
        return queries

    def write_create_sql_script(self, script_path):
        """
        writes all queries needed to install the extensions UDF'S to the script_path.
        """
        install_scripts_constants = self.get_constant_set()
        queries = {}

        for constant_set in install_scripts_constants:
            queries = queries | self.make_script_deployment_queries(constant_set)

        with open(script_path, "w") as create_script:
            create_script.write(
                "-- this script is created automatically. Call 'write_create_script' if you need to update it.\n\n"
            )

            for query in queries.values():
                # Write the new data to the file
                create_script.write(query)
                create_script.write("\n")
