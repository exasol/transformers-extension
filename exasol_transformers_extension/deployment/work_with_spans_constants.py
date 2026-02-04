"""Constants defining which UDF to install if work_with_spans=true"""

import pathlib

from exasol_transformers_extension.deployment.install_scripts_constants import (
    InstallScriptsConstants,
)

UDF_CALL_TEMPLATES = {
    "span_ai_extract_extended_udf_call.py": "span_ai_extract_extended_udf.jinja.sql",
    "span_ai_classify_extended_udf_call.py": "span_ai_classify_extended_udf.jinja.sql",
}

work_with_spans_constants = InstallScriptsConstants(
    base_dir="exasol_transformers_extension",
    templates_dir=pathlib.Path("resources", "templates", "with_spans"),
    udf_callers_dir_suffix="udfs.callers.with_spans",
    udf_callers_templates=UDF_CALL_TEMPLATES,
    ordered_columns=["model_name", "bucketfs_conn", "sub_dir"],
)
