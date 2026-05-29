from test.integration_tests.without_db.udfs.utils.matcher import (
    ColumnsMatcher,
    NoErrorMessageMatcher,
    Result,
)
from test.integration_tests.without_db.udfs.utils.mock_context import MockContext
from test.integration_tests.without_db.udfs.utils.mock_exa_environment import (
    MockExaEnvironment,
)
from test.utils.mock_connections import create_mounted_bucketfs_connection
from test.utils.parameters import model_params

import pandas as pd
import pytest
import torch

from exasol_transformers_extension.deployment.default_udf_parameters import (
    DEFAULT_BUCKETFS_CONN_NAME,
)
from exasol_transformers_extension.udfs.models.ai_extract_entities_udf import (
    AiExtractEntitiesUDF,
)


@pytest.mark.parametrize(
    "description, n_rows",
    [
        ("batch input", 3),
        ("single input", 1),
    ],
)
def test_ai_extract_entities_udf(
    description,
    n_rows,
    prepare_default_token_classification_model_for_local_bucketfs,
):

    bucketfs_base_path = prepare_default_token_classification_model_for_local_bucketfs
    bucketfs_connection = create_mounted_bucketfs_connection(bucketfs_base_path)

    batch_size = 2
    sample_data = [(model_params.text_data + str(i),) for i in range(n_rows)]
    columns = [
        "text_data",
    ]

    sample_df = pd.DataFrame(data=sample_data, columns=columns)
    ctx = MockContext(input_df=sample_df)
    exa = MockExaEnvironment({DEFAULT_BUCKETFS_CONN_NAME: bucketfs_connection})

    extractor = AiExtractEntitiesUDF(exa, batch_size=batch_size)
    extractor.run(ctx)

    result_dfs = ctx.get_emitted()
    result_df = pd.concat(result_dfs)
    new_columns = [
        "start_pos",
        "end_pos",
        "word",
        "entity",
        "score",
        "error_message",
    ]

    result = Result(result_df)
    assert (
        result == ColumnsMatcher(columns=columns, new_columns=new_columns)
        and result == NoErrorMessageMatcher()
    )
