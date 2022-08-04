import pathlib
from importlib_resources import files

BASE_DIR = "exasol_transformers_extension"
TEMPLATES_DIR = pathlib.Path("resources", "templates")
SOURCE_DIR = files(f"{BASE_DIR}.udfs")

UDF_CALL_TEMPLATES = {
    "model_downloader_udf_call.py":
        "model_downloader_udf.jinja.sql",
    "sequence_classification_single_text_udf_call.py":
        "sequence_classification_single_text_udf.jinja.sql",
    "sequence_classification_text_pair_udf_call.py":
        "sequence_classification_text_pair_udf.jinja.sql",
    "question_answering_udf_call.py":
        "question_answering_udf.jinja.sql",
}

ORDERED_COLUMNS = ['model_name', 'bucketfs_conn', 'sub_dir']
