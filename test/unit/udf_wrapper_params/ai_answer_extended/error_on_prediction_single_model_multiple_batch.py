from pathlib import PurePosixPath

from exasol_udf_mock_python.connection import Connection


class ErrorOnPredictionSingleModelMultipleBatch:
    """
    error on prediction, single model, multiple batch,
    """

    expected_model_counter = 1
    batch_size = 2
    data_size = 4

    input_data = [
        (None, "bfs_conn1", "sub_dir1", "model1", "question", "error on pred")
    ] * data_size
    expected_output_data = [
        (
            "bfs_conn1",
            "sub_dir1",
            "model1",
            "question",
            "error on pred",
            None,
            "Traceback",
        )
    ] * data_size

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, "bfs_conn1")
    bfs_connections = {"bfs_conn1": Connection(address=f"file://{base_cache_dir1}")}

    model_output_df_model1 = [[[{None}]] * data_size]

    models_output_df = [model_output_df_model1]
