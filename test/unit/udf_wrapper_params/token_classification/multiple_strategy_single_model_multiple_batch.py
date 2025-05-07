import dataclasses
from pathlib import PurePosixPath
from test.unit.udf_wrapper_params.token_classification.make_data_row_functions import (
    bucketfs_conn,
    make_input_row,
    make_input_row_with_span,
    make_model_output_for_one_input_row,
    make_output_row,
    make_output_row_with_span,
)

from exasol_udf_mock_python.connection import Connection


@dataclasses.dataclass
class MultipleStrategySingleModelNameMultipleBatch:
    """
    Multiple strategies, single model, single batch
    """

    expected_model_counter = 1
    batch_size = 2
    data_size = 2
    agg_strategy_null = None
    agg_strategy_none = "none"
    agg_strategy_simple = "simple"
    n_entities = 3

    input_data = (
        make_input_row(aggregation_strategy=agg_strategy_null) * data_size
        + make_input_row(aggregation_strategy=agg_strategy_none) * data_size
        + make_input_row(aggregation_strategy=agg_strategy_simple) * data_size
    )
    output_data = (
        make_output_row(aggregation_strategy=agg_strategy_simple)
        * n_entities
        * data_size
        + make_output_row(aggregation_strategy=agg_strategy_none)
        * n_entities
        * data_size
        + make_output_row(aggregation_strategy=agg_strategy_simple)
        * n_entities
        * data_size
    )

    work_with_span_input_data = (
        make_input_row_with_span(aggregation_strategy=agg_strategy_null) * data_size
        + make_input_row_with_span(aggregation_strategy=agg_strategy_none) * data_size
        + make_input_row_with_span(aggregation_strategy=agg_strategy_simple) * data_size
    )
    work_with_span_output_data = (
        make_output_row_with_span(aggregation_strategy=agg_strategy_simple)
        * n_entities
        * data_size
        + make_output_row_with_span(aggregation_strategy=agg_strategy_none)
        * n_entities
        * data_size
        + make_output_row_with_span(aggregation_strategy=agg_strategy_simple)
        * n_entities
        * data_size
    )
    # data gets divided into 3 batches
    tokenizer_model_output_df_model1 = (
        [
            make_model_output_for_one_input_row(
                number_entities=n_entities, aggregation_strategy=agg_strategy_null
            )
            * data_size
        ]
        + [
            make_model_output_for_one_input_row(
                number_entities=n_entities, aggregation_strategy=agg_strategy_none
            )
            * data_size
        ]
        + [
            make_model_output_for_one_input_row(
                number_entities=n_entities, aggregation_strategy=agg_strategy_simple
            )
            * data_size
        ]
    )
    tokenizer_models_output_df = [tokenizer_model_output_df_model1]

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, bucketfs_conn)
    bfs_connections = {bucketfs_conn: Connection(address=f"file://{base_cache_dir1}")}
