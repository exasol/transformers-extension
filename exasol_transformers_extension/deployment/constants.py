from pathlib import Path

from exasol_transformers_extension.deployment.install_scripts_constants import InstallScriptsConstants
# todo change in docu
UDF_CALL_TEMPLATES = {
    "model_downloader_udf_call.py":
        "model_downloader_udf.jinja.sql",
    "sequence_classification_single_text_udf_call.py":
        "sequence_classification_single_text_udf.jinja.sql",
    "sequence_classification_text_pair_udf_call.py":
        "sequence_classification_text_pair_udf.jinja.sql",
    "question_answering_udf_call.py":
        "question_answering_udf.jinja.sql",
    "filling_mask_udf_call.py":
        "filling_mask_udf.jinja.sql",
    "text_generation_udf_call.py":
        "text_generation_udf.jinja.sql",
    "translation_udf_call.py":
        "translation_udf.jinja.sql",
    "zero_shot_text_classification_udf.py":
        "zero_shot_text_classification_udf.jinja.sql"
}

constants = InstallScriptsConstants(
    base_dir="exasol_transformers_extension",
    templates_dir=Path("resources", "templates"),
    udf_callers_dir_suffix="udfs.callers",
    udf_callers_templates=UDF_CALL_TEMPLATES,
    ordered_columns=['model_name', 'bucketfs_conn', 'sub_dir']
)