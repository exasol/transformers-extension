"""Constants defining which UDF to install if work_with_spans=false"""

import pathlib

from exasol_transformers_extension.deployment.install_scripts_constants import (
    InstallScriptsConstants,
)

UDF_CALL_TEMPLATES = {
    "ai_extract_extended_udf_call.py": "ai_extract_extended_udf.jinja.sql",
    "ai_classify_extended_udf.py": "ai_classify_extended_udf.jinja.sql",
}

work_without_spans_constants = InstallScriptsConstants(
    base_dir="exasol_transformers_extension",
    templates_dir=pathlib.Path("resources", "templates", "without_spans"),
    udf_callers_dir_suffix="udfs.callers",
    udf_callers_templates=UDF_CALL_TEMPLATES,
    ordered_columns=["model_name", "bucketfs_conn", "sub_dir"],
)
