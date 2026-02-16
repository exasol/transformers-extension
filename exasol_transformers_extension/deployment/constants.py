"""Constants defining which UDF to install independent fo work_with_spans"""

from pathlib import Path

from exasol_transformers_extension.deployment.install_scripts_constants import (
    InstallScriptsConstants,
)

UDF_CALL_TEMPLATES = {
    "model_downloader_udf_call.py": "model_downloader_udf.jinja.sql",
    "install_default_models_call.py": "install_default_models_udf.jinja.sql",
    "ls_models_udf.py": "ls_models_udf.jinja.sql",
    "ai_custom_classify_extended_udf_call.py": "ai_custom_classify_extended_udf.jinja.sql",
    "ai_entailment_extended_udf_call.py": "ai_entailment_extended_udf.jinja.sql",
    "ai_answer_extended_udf_call.py": "ai_answer_extended_udf.jinja.sql",
    "ai_fill_mask_extended_udf_call.py": "ai_fill_mask_extended_udf.sql",
    "ai_complete_extended_udf_call.py": "ai_complete_extended_udf.jinja.sql",
    "ai_translate_extended_udf_call.py": "ai_translate_extended_udf.jinja.sql",
    "delete_model_udf_call.py": "delete_model_udf.jinja.sql",
}

constants = InstallScriptsConstants(
    base_dir="exasol_transformers_extension",
    templates_dir=Path("resources", "templates"),
    udf_callers_dir_suffix="udfs.callers",
    udf_callers_templates=UDF_CALL_TEMPLATES,
    ordered_columns=["model_name", "bucketfs_conn", "sub_dir"],
)
