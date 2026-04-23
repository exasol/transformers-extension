from pathlib import PurePosixPath

from exasol_transformers_extension.deployment.default_udf_parameters import DEFAULT_BUCKETFS_CONN_NAME
from test.unit.udf_wrapper_params.text_classification.make_data_row_functions import (
    make_model_output_for_one_input_row,
    LABEL_SCORES,
)

from exasol_udf_mock_python.connection import Connection


class DefaultValuesMultipleBatchComplete:
    """
    multiple model, multiple batch, last batch complete, return_ranks HIGHEST
    """
    expected_model_counter = 1
    batch_size = 2
    data_size = 2

    text_data = "My test text"
    error_msg = None

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, DEFAULT_BUCKETFS_CONN_NAME)
    bfs_connections = {
        DEFAULT_BUCKETFS_CONN_NAME: Connection(address=f"file://{base_cache_dir1}")
    }

    inputs_single_text = (
        [(text_data,)]
        * data_size
        + [(text_data,)]
        * data_size
    )

    output_single_text_model =  [
            (
                text_data,
                LABEL_SCORES.label_scores[3].label,
                LABEL_SCORES.label_scores[3].score,
                error_msg,
            )
        ]#todo


    outputs_single_text = (
        output_single_text_model * data_size + output_single_text_model * data_size
    )

    model_output_single_text_model = [
        make_model_output_for_one_input_row() * data_size + make_model_output_for_one_input_row() * data_size
    ]

    text_class_models_output_df_single_text = [
        model_output_single_text_model
    ]
