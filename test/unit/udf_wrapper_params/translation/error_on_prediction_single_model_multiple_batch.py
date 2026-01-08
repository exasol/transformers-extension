from pathlib import PurePosixPath
import dataclasses

from exasol_udf_mock_python.connection import Connection

from test.unit.udf_wrapper_params.translation.make_data_row_functions import (
    bucketfs_conn,
    make_input_row,
    make_model_output_for_one_input_row,
    make_udf_output_for_one_input_row, sub_dir, model_name, text_data, translation_text,
)
from test.unit.utils.utils_for_udf_tests import make_number_of_strings

@dataclasses.dataclass
class ErrorOnPredictionSingleModelMultipleBatch:
    """
    not cached error, single model, multiple batch
    """

    expected_model_counter = 1
    batch_size = 2
    data_size = 5

    input_data = (make_input_row(text_data= "error on pred") * data_size)

    output_data = (make_udf_output_for_one_input_row(text_data= "error on pred",
                                                     translation_text=None,
                                                     error_msg="Traceback") * data_size)

    translation_models_output_df = [
        [([{"translation_text": None}] * batch_size)],
        [([{"translation_text": None}] * batch_size)],
        [([{"translation_text": None}])]#todo error in model output?
    ]

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir = PurePosixPath(tmpdir_name, bucketfs_conn)
    bfs_connections = {
        bucketfs_conn: Connection(address=f"file://{base_cache_dir}"),
    }