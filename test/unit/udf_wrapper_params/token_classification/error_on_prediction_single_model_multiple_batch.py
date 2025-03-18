import dataclasses
from pathlib import PurePosixPath
from test.unit.udf_wrapper_params.token_classification.make_data_row_functions import make_input_row, \
    make_output_row, make_input_row_with_span, bucketfs_conn, \
    text_doc_id, text_start, text_end, agg_strategy_simple, sub_dir, model_name

from exasol_udf_mock_python.connection import Connection

@dataclasses.dataclass
class ErrorOnPredictionSingleModelMultipleBatch:
    """
    error on prediction, single model, multiple batch,
    """
    expected_model_counter = 1
    batch_size = 2
    data_size = 5
    n_entities = 3

    input_data = make_input_row(text_data="error on pred") * data_size
    output_data = make_output_row(text_data="error on pred",
                                  score=None, start=None, end=None, word=None, entity=None,
                                  error_msg="Traceback") * 1 * data_size #error on pred -> only one output per input

    work_with_span_input_data = make_input_row_with_span(text_data="error on pred") * data_size
    work_with_span_output_data =  [(bucketfs_conn, sub_dir, model_name,
                                text_doc_id, text_start, text_end, agg_strategy_simple,
                                None, None, None, text_doc_id, None, None,
                                "Traceback")] * 1 * data_size #error on pred -> only one output per input


    number_complete_batches = data_size // batch_size
    number_remaining_data_entries_in_last_batch = data_size % batch_size
    # error on pred -> only one output per input
    tokenizer_model_output_df_model1 = [[Exception("Traceback mock_pipeline is throwing an error intentionally")]
                                        * batch_size] * \
                                       number_complete_batches + \
                                       [[Exception("Traceback mock_pipeline is throwing an error intentionally")] *
                                        number_remaining_data_entries_in_last_batch]
    tokenizer_models_output_df = [tokenizer_model_output_df_model1]

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, bucketfs_conn)
    bfs_connections = {
        bucketfs_conn: Connection(address=f"file://{base_cache_dir1}")
    }
