from pathlib import PurePosixPath
from test.unit.udf_wrapper_params.ai_extract_entities.make_data_row_functions import (
    make_model_output_for_one_input_row,
    text_data, make_output_row,
)

from exasol_udf_mock_python.connection import Connection
from exasol_transformers_extension.deployment.default_udf_parameters import (
    DEFAULT_BUCKETFS_CONN_NAME,
)


class DefaultValuesMultipleBatchComplete:
    """
    single model, multiple batch, default model settings
    """

    expected_model_counter = 1
    batch_size = 2
    data_size = 2
    n_entities = 3

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, DEFAULT_BUCKETFS_CONN_NAME)
    bfs_connections = {
        DEFAULT_BUCKETFS_CONN_NAME: Connection(address=f"file://{base_cache_dir1}")
    }

    inputs = [(text_data,)] * data_size + [(text_data,)] * data_size

    output_1_input_row = make_output_row()

    outputs = (
        output_1_input_row * n_entities * data_size
        + output_1_input_row * n_entities * data_size
    )

    model_output_1_batch = (
        make_model_output_for_one_input_row(number_entities=n_entities) * data_size
    )
    model_output_2_batch = (
        make_model_output_for_one_input_row(number_entities=n_entities) * data_size
    )
    tokenizer_models_output_df = [[model_output_1_batch, model_output_2_batch]]
