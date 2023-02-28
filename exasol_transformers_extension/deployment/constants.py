import pathlib
from importlib_resources import files

BASE_DIR = "exasol_transformers_extension"
TEMPLATES_DIR = pathlib.Path("resources", "templates")
UDF_CALLERS_DIR = files(f"{BASE_DIR}.udfs.callers")

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
    "token_classification_udf_call.py":
        "token_classification_udf.jinja.sql",
    "translation_udf_call.py":
        "translation_udf.jinja.sql",
    "zero_shot_text_classification_udf.py":
        "zero_shot_text_classification_udf.jinja.sql"
}

ORDERED_COLUMNS = ['model_name', 'bucketfs_conn', 'sub_dir']
