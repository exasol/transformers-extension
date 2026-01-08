from pathlib import PurePosixPath
import dataclasses

from exasol_udf_mock_python.connection import Connection
from test.unit.udf_wrapper_params.translation.make_data_row_functions import (
    bucketfs_conn,
    make_input_row,
    make_model_output_for_one_input_row,
    make_udf_output_for_one_input_row, max_length,
)
@dataclasses.dataclass
class MultipleMaxLengthSingleModelNameSingleBatch:
    """
    multiple max_length, single model, single batch
    """

    expected_model_counter = 1
    batch_size = 4
    data_size = 2
    max_length1 = max_length
    max_length2 = 2

    input_data = (make_input_row(max_length=max_length1) * data_size +
                  make_input_row(max_length=max_length2) * data_size)

    output_data = (make_udf_output_for_one_input_row(max_length=max_length1) * data_size +
                   make_udf_output_for_one_input_row(max_length=max_length2) * data_size)

    translation_models_output_df = [
        (make_model_output_for_one_input_row(max_length=max_length1) * data_size) +
        (make_model_output_for_one_input_row(max_length=max_length2) * data_size),
    ]

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir = PurePosixPath(tmpdir_name, bucketfs_conn)
    bfs_connections = {
        bucketfs_conn: Connection(address=f"file://{base_cache_dir}"),
    }
