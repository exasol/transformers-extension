import dataclasses
from pathlib import PurePosixPath
from test.unit.udf_wrapper_params.ai_classify_extended.make_data_row_functions import (
    bucketfs_conn,
    make_input_row,
    make_input_row_with_span,
    make_model_output_for_one_input_row,
    make_udf_output_for_one_input_row_with_span,
    make_udf_output_for_one_input_row_without_span,
)

from exasol_udf_mock_python.connection import Connection


@dataclasses.dataclass
class ReturnAllMultipleLabelsSingleModelMultipleBatch:
    """
    return_ranks ALL, multiple labels, single model, multiple batches
    """

    expected_model_counter = 1
    batch_size = 2
    data_size = 4

    input_data = make_input_row() * batch_size + make_input_row() * batch_size

    output_data = make_udf_output_for_one_input_row_without_span() * data_size

    zero_shot_model_output_df_batch1 = (
        make_model_output_for_one_input_row() * batch_size
    )

    work_with_span_input_data = make_input_row_with_span() * data_size

    work_with_span_output_data = (
        make_udf_output_for_one_input_row_with_span() * data_size
    )

    zero_shot_models_output_df = [
        [
            zero_shot_model_output_df_batch1,
            zero_shot_model_output_df_batch1,
        ]
    ]

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir = PurePosixPath(tmpdir_name, bucketfs_conn)
    bfs_connections = {
        bucketfs_conn: Connection(address=f"file://{base_cache_dir}"),
    }
