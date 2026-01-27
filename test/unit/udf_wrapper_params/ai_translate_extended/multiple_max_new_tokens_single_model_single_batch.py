import dataclasses
from pathlib import PurePosixPath
from test.unit.udf_wrapper_params.ai_translate_extended.make_data_row_functions import (
    bucketfs_conn,
    make_input_row,
    make_model_output_for_one_input_row,
    make_udf_output_for_one_input_row,
    max_new_tokens,
)

from exasol_udf_mock_python.connection import Connection


@dataclasses.dataclass
class MultipleMaxLengthSingleModelNameSingleBatch:
    """
    multiple max_new_tokens, single model, single batch
    """

    expected_model_counter = 1
    batch_size = 4
    data_size = 2
    max_new_tokens1 = max_new_tokens
    max_new_tokens2 = 2

    input_data = (
        make_input_row(max_new_tokens=max_new_tokens1) * data_size
        + make_input_row(max_new_tokens=max_new_tokens2) * data_size
    )

    output_data = (
        make_udf_output_for_one_input_row(max_new_tokens=max_new_tokens1) * data_size
        + make_udf_output_for_one_input_row(max_new_tokens=max_new_tokens2) * data_size
    )

    translation_model_output_df_maxlen1 = [
        make_model_output_for_one_input_row(max_new_tokens=max_new_tokens1) * data_size
    ]
    translation_model_output_df_maxlen2 = [
        make_model_output_for_one_input_row(max_new_tokens=max_new_tokens2) * data_size
    ]

    translation_models_output_df = [
        translation_model_output_df_maxlen1 + translation_model_output_df_maxlen2
    ]

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir = PurePosixPath(tmpdir_name, bucketfs_conn)
    bfs_connections = {
        bucketfs_conn: Connection(address=f"file://{base_cache_dir}"),
    }
