from pathlib import PurePosixPath
from test.unit.udf_wrapper_params.ai_classify_extended.make_data_row_functions import (
    LABEL_SCORES,
    candidate_labels,
    make_model_output_for_one_input_row,
)

from exasol_udf_mock_python.connection import Connection

from exasol_transformers_extension.deployment.default_udf_parameters import (
    DEFAULT_BUCKETFS_CONN_NAME,
)


class DefaultValuesMultipleBatchComplete:
    expected_model_counter = 1
    batch_size = 2
    data_size = 2
    text_data = "My test text"
    candidate_labels_str = ",".join(candidate_labels)
    error_msg = None

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, DEFAULT_BUCKETFS_CONN_NAME)
    bfs_connections = {
        DEFAULT_BUCKETFS_CONN_NAME: Connection(address=f"file://{base_cache_dir1}")
    }

    inputs = [
        (text_data, candidate_labels_str),
        (text_data, candidate_labels_str),
    ]

    output_row = (
        text_data,
        candidate_labels_str,
        LABEL_SCORES.label_scores[3].label,
        LABEL_SCORES.label_scores[3].score,
        error_msg,
    )
    outputs_single_text = [output_row, output_row]

    text_class_models_output_df = [
        [make_model_output_for_one_input_row() * data_size],
    ]
