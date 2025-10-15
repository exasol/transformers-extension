from test.integration_tests.without_db.udfs.utils.matcher import (
    ErrorMessageMatcher,
    NewColumnsEmptyMatcher,
    NoErrorMessageMatcher,
    Result,
    ShapeMatcher,
)
from test.integration_tests.without_db.udfs.utils.mock_context import MockContext
from test.integration_tests.without_db.udfs.utils.mock_exa_environment import MockExaEnvironment
from test.utils.mock_connections import create_mounted_bucketfs_connection
from test.utils.parameters import model_params

import pandas as pd
import pytest
import torch
from exasol_udf_mock_python.connection import Connection

from exasol_transformers_extension.udfs.models.translation_udf import TranslationUDF


@pytest.mark.parametrize(
    "description,  device_id, languages",
    [
        ("on CPU with single input", None, [("English", "French")]),
        (
            "on CPU with batch input, single-pair language",
            None,
            [("English", "French")] * 3,
        ),
        (
            "on CPU with batch input, multi language",
            None,
            [("English", "French"), ("English", "German"), ("English", "Romanian")],
        ),
        ("on GPU with single input", 0, [("English", "French")]),
        (
            "on GPU with batch input, single-pair language",
            0,
            [("English", "French")] * 3,
        ),
        (
            "on GPU with batch input, multi language",
            0,
            [("English", "French"), ("English", "German"), ("English", "Romanian")],
        ),
    ],
)
def test_translation_udf(
    description, device_id, languages, prepare_translation_model_for_local_bucketfs
):
    if device_id is not None and not torch.cuda.is_available():
        pytest.skip(
            f"There is no available device({device_id}) " f"to execute the test"
        )

    bucketfs_base_path = prepare_translation_model_for_local_bucketfs
    bucketfs_conn_name = "bucketfs_connection"
    bucketfs_connection = create_mounted_bucketfs_connection(bucketfs_base_path)

    batch_size = 2
    sample_data = [
        (
            None,
            bucketfs_conn_name,
            model_params.sub_dir,
            model_params.seq2seq_model_specs.model_name,
            model_params.text_data,
            src_lang,
            target_lang,
            50,
        )
        for src_lang, target_lang in languages
    ]
    columns = [
        "device_id",
        "bucketfs_conn",
        "sub_dir",
        "model_name",
        "text_data",
        "source_language",
        "target_language",
        "max_length",
    ]

    sample_df = pd.DataFrame(data=sample_data, columns=columns)
    ctx = MockContext(input_df=sample_df)
    exa = MockExaEnvironment({bucketfs_conn_name: bucketfs_connection})

    sequence_classifier = TranslationUDF(exa, batch_size=batch_size)
    sequence_classifier.run(ctx)

    result_df = ctx.get_emitted()[0][0]
    new_columns = ["translation_text", "error_message"]

    result = Result(result_df)
    assert (
        result
        == ShapeMatcher(columns=columns, new_columns=new_columns, n_rows=len(languages))
        and result == NoErrorMessageMatcher()
    )


@pytest.mark.parametrize(
    "description,  device_id, languages",
    [
        ("on CPU with single input", None, [("English", "French")]),
        (
            "on CPU with batch input, single-pair language",
            None,
            [("English", "French")] * 3,
        ),
        (
            "on CPU with batch input, multi language",
            None,
            [("English", "French"), ("English", "German"), ("English", "Romanian")],
        ),
        ("on GPU with single input", 0, [("English", "French")]),
        (
            "on GPU with batch input, single-pair language",
            0,
            [("English", "French")] * 3,
        ),
        (
            "on GPU with batch input, multi language",
            0,
            [("English", "French"), ("English", "German"), ("English", "Romanian")],
        ),
    ],
)
def test_translation_udf_on_error_handling(
    description, device_id, languages, prepare_translation_model_for_local_bucketfs
):
    if device_id is not None and not torch.cuda.is_available():
        pytest.skip(
            f"There is no available device({device_id}) " f"to execute the test"
        )

    bucketfs_base_path = prepare_translation_model_for_local_bucketfs
    bucketfs_conn_name = "bucketfs_connection"
    bucketfs_connection = Connection(address=f"file://{bucketfs_base_path}")

    batch_size = 2
    sample_data = [
        (
            None,
            bucketfs_conn_name,
            model_params.sub_dir,
            "not existing model",
            model_params.text_data,
            src_lang,
            target_lang,
            50,
        )
        for src_lang, target_lang in languages
    ]
    columns = [
        "device_id",
        "bucketfs_conn",
        "sub_dir",
        "model_name",
        "text_data",
        "source_language",
        "target_language",
        "max_length",
    ]

    sample_df = pd.DataFrame(data=sample_data, columns=columns)
    ctx = MockContext(input_df=sample_df)
    exa = MockExaEnvironment({bucketfs_conn_name: bucketfs_connection})

    sequence_classifier = TranslationUDF(exa, batch_size=batch_size)
    sequence_classifier.run(ctx)

    result_df = ctx.get_emitted()[0][0]
    new_columns = ["translation_text", "error_message"]

    result = Result(result_df)
    assert (
        result
        == ShapeMatcher(columns=columns, new_columns=new_columns, n_rows=len(languages))
        and result == NewColumnsEmptyMatcher(new_columns=new_columns)
        and result == ErrorMessageMatcher()
    )
