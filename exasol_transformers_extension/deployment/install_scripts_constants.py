"""Class InstallScriptsConstants to encapsulate different sets of constants for each set of UDFs to be installed."""

from pathlib import Path

from importlib_resources import files


class InstallScriptsConstants:
    """
    Class to encapsulate different sets of constants for each set of UDFs/scripts to be installed.
    """

    def __init__(
        self,
        base_dir: str,
        templates_dir: Path,
        udf_callers_dir_suffix: str,
        udf_callers_templates: dict,
        ordered_columns: list,
    ):
        """

        :param base_dir:
                base directory to install scripts from, usually "exasol_transformers_extension"
        :param templates_dir:
                directory containing udf templates, excluding base_dir. e.g "resources/templates"
        :param udf_callers_dir_suffix:
                suffix from where to import callers from. e.g "udfs.callers"
        :param udf_callers_templates:
                dict containing pairs of caller python files and jinja templates.
                the caller python file must be found in base_dir/udf_callers_dir_suffix
                (if udf_callers_dir_suffix was converted to a path)
                the jinja template must be found in base_dir/templates_dir
                a key value pair may look like eg: "token_classification_udf_call.py":
                    "token_classification_udf.jinja.sql"
        :param ordered_columns:
                list of ordered columns the udfs designated by the instance of InstallScriptsConstants have in common.
                e.g. ['model_name', 'bucketfs_conn', 'sub_dir']

        """

        self.base_dir = base_dir
        self.templates_dir = templates_dir
        self.udf_callers_dir = files(f"{self.base_dir}.{udf_callers_dir_suffix}")
        self.udf_callers_templates = udf_callers_templates
        self.ordered_columns = ordered_columns
