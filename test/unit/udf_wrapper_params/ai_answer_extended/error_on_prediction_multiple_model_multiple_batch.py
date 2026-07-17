from pathlib import PurePosixPath

from exasol_udf_mock_python.connection import Connection


class ErrorOnPredictionMultipleModelMultipleBatch:
    """
    not cached error, multiple model, multiple batch
    """

    expected_model_counter = 2
    batch_size = 3
    data_size = 2

    input_data = [
        (None, "bfs_conn1", "sub_dir1", "model1", "question", "context")
    ] * data_size + [
        (None, "bfs_conn2", "sub_dir2", "model2", "question", "error on pred")
    ] * data_size
    expected_output_data = [
        (
            "bfs_conn1",
            "sub_dir1",
            "model1",
            "question",
            "context",
            "answer 1",
            None,
        )
    ] * data_size + [
        (
            "bfs_conn2",
            "sub_dir2",
            "model2",
            "question",
            "error on pred",
            None,
            "Traceback",
        )
    ] * data_size

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, "bfs_conn1")
    base_cache_dir2 = PurePosixPath(tmpdir_name, "bfs_conn2")

    bfs_connections = {
        "bfs_conn1": Connection(address=f"file://{base_cache_dir1}"),
        "bfs_conn2": Connection(address=f"file://{base_cache_dir2}"),
    }

    model_output_df_model1 = [[[{"generated_text": "answer 1"}]] * data_size]
    model_output_df_model2 = [[[{None}]] * data_size]
    models_output_df = [model_output_df_model1, model_output_df_model2]
