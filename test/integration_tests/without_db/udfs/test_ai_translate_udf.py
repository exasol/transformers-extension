from exasol_transformers_extension.deployment.default_udf_parameters import DEFAULT_BUCKETFS_CONN_NAME
from exasol_transformers_extension.udfs.models.ai_translate import AiTranslateUDF
from test.integration_tests.without_db.udfs.utils.matcher import (
    NoErrorMessageMatcher,
    Result,
    ShapeMatcher,
)
from test.integration_tests.without_db.udfs.utils.mock_context import MockContext
from test.integration_tests.without_db.udfs.utils.mock_exa_environment import (
    MockExaEnvironment,
)
from test.utils.mock_connections import create_mounted_bucketfs_connection
from test.utils.parameters import model_params

import pandas as pd
import pytest


@pytest.mark.parametrize(
    "description, languages",
    [
        ("on CPU with single input", [("English", "French")]),
        (
            "on CPU with batch input, single-pair language",
            [("English", "French")],
        ),
        (
            "on CPU with batch input, multi language",
            [("English", "French"), ("English", "German"), ("English", "Romanian")],
        ),
    ],
)
def test_ai_translate_extended_udf(
    description, languages, prepare_default_translation_model_for_local_bucketfs
):
    bucketfs_base_path = prepare_default_translation_model_for_local_bucketfs
    bucketfs_connection = create_mounted_bucketfs_connection(bucketfs_base_path)

    batch_size = 2

    sample_data = [
        (
            model_params.text_data,
            src_lang,
            target_lang,
        )
        for src_lang, target_lang in languages
    ]

    columns = [
        "text_data",
        "source_language",
        "target_language",
    ]

    sample_df = pd.DataFrame(data=sample_data, columns=columns)
    ctx = MockContext(input_df=sample_df)
    exa = MockExaEnvironment({DEFAULT_BUCKETFS_CONN_NAME: bucketfs_connection})

    sequence_classifier = AiTranslateUDF(exa, batch_size=batch_size)
    sequence_classifier.run(ctx)

    result_dfs = ctx.get_emitted()
    result_df = pd.concat(result_dfs)
    new_columns = ["translation_text", "error_message"]

    result = Result(result_df)
    assert (
        result
        == ShapeMatcher(
            columns=columns,
            new_columns=new_columns,
            n_rows=len(languages),
            removed_columns=[],
        )
        and result == NoErrorMessageMatcher()
    )

