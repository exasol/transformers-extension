import pathlib
from importlib_resources import files

BASE_DIR = "exasol_transformers_extension"
TEMPLATES_DIR = pathlib.Path("resources", "templates")
SOURCE_DIR = files(f"{BASE_DIR}.udfs")

UDF_CALL_TEMPLATES = {
    "model_downloader_udf_call.py":
        "model_downloader_udf.jinja.sql",
    "sequence_classification_single_text_udf.py":
        "sequence_classification_single_text_udf.jinja.sql",
}


