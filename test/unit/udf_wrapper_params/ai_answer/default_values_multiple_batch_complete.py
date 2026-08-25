from pathlib import PurePosixPath

from exasol_udf_mock_python.connection import Connection

from exasol_transformers_extension.deployment.default_udf_parameters import (
    DEFAULT_BUCKETFS_CONN_NAME,
)


class DefaultValuesMultipleBatchComplete:

    expected_model_counter = 1
    batch_size = 2
    data_size = 4

    input_data = [
        ("question", "context"),
    ] * data_size

    expected_output_data = [
        (
            "question",
            "context",
            "answer",
            None,
        )
    ] * data_size

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir = PurePosixPath(tmpdir_name, DEFAULT_BUCKETFS_CONN_NAME)

    bfs_connections = {
        DEFAULT_BUCKETFS_CONN_NAME: Connection(address=f"file://{base_cache_dir}"),
    }

    model_output_df_1_batch = [[{"generated_text": "answer"}]] * batch_size
    model_output_df_2_batch = [[{"generated_text": "answer"}]] * batch_size
    model_output_dfs = [[model_output_df_1_batch, model_output_df_2_batch]]
