from pathlib import PurePosixPath
from exasol_udf_mock_python.connection import Connection
from tests.unit_tests.udf_wrapper_params.token_classification.make_data_row_functions import make_input_row, \
    make_output_row, make_input_row_with_span, make_output_row_with_span, bucketfs_conn, \
    make_model_output_for_one_input_row


class MultipleStrategySingleModelNameSingleBatch:
    """
    multiple strategy, single model, single batch
    """
    expected_model_counter = 1
    batch_size = 6
    data_size = 2
    agg_strategy_null = None
    agg_strategy_none = "none"
    agg_strategy_simple = "simple"
    n_entities = 3

    input_data = make_input_row(aggregation_strategy=agg_strategy_null) * data_size + \
                 make_input_row(aggregation_strategy=agg_strategy_none) * data_size + \
                 make_input_row(aggregation_strategy=agg_strategy_simple) * data_size
    output_data = make_output_row(aggregation_strategy=agg_strategy_simple) * n_entities * data_size + \
                  make_output_row(aggregation_strategy=agg_strategy_simple) * n_entities * data_size + \
                  make_output_row(aggregation_strategy=agg_strategy_none) * n_entities * data_size

    work_with_span_input_data = make_input_row_with_span(aggregation_strategy=agg_strategy_null) * data_size + \
                                make_input_row_with_span(aggregation_strategy=agg_strategy_none) * data_size + \
                                make_input_row_with_span(aggregation_strategy=agg_strategy_simple) * data_size
    work_with_span_output_data = make_output_row_with_span(aggregation_strategy=agg_strategy_simple) * n_entities * data_size + \
                                 make_output_row_with_span(aggregation_strategy=agg_strategy_simple) * n_entities * data_size + \
                                 make_output_row_with_span(aggregation_strategy=agg_strategy_none) * n_entities * data_size
    # data gets divided into one batch with agg strategy simple, 4 input rows, second batch agg strategy none 2 input rows
    tokenizer_model_output_df = [make_model_output_for_one_input_row(number_entities=n_entities) * data_size * 2] + \
                                [make_model_output_for_one_input_row(number_entities=n_entities) * data_size]


    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, bucketfs_conn)
    bfs_connections = {
        bucketfs_conn: Connection(address=f"file://{base_cache_dir1}")
    }

