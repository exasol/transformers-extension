import dataclasses
from pathlib import PurePosixPath
from test.unit.udf_wrapper_params.zero_shot.make_data_row_functions import (
    bucketfs_conn,
    make_input_row,
    make_input_row_with_span,
    make_model_output_for_one_input_row,
    make_udf_output_for_one_input_row_without_span,
    make_udf_output_for_one_input_row_with_span,
)

from exasol_udf_mock_python.connection import Connection


@dataclasses.dataclass
class ReturnAllMultipleLabelsSingleModelSingleBatch:
    """
    Multiple labels, single model, single batch, batch complete
    """

    expected_model_counter = 1
    batch_size = 1
    data_size = 1

    input_data = make_input_row() * data_size


    output_data = make_udf_output_for_one_input_row_without_span() * data_size

    zero_shot_models_output_df = [[
        make_model_output_for_one_input_row() * data_size,
    ]]

    work_with_span_input_data = (
        make_input_row_with_span() * data_size
    )

    work_with_span_output_data = (
        (make_udf_output_for_one_input_row_with_span()) * data_size
    )

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, bucketfs_conn)
    bfs_connections = {
        bucketfs_conn: Connection(address=f"file://{base_cache_dir1}"),
    }
