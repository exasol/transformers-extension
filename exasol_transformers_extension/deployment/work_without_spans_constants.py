import pathlib
from importlib_resources import files

BASE_DIR = "exasol_transformers_extension"
TEMPLATES_DIR = pathlib.Path("resources", "templates", "without_spans")
UDF_CALLERS_DIR = files(f"{BASE_DIR}.udfs.callers")

UDF_CALL_TEMPLATES = {
    "token_classification_udf_call.py":
        "token_classification_udf.jinja.sql",
}

ORDERED_COLUMNS = ['model_name', 'bucketfs_conn', 'sub_dir']