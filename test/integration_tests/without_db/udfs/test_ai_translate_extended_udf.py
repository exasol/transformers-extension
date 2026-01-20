from test.integration_tests.without_db.udfs.utils.matcher import (
    ErrorMessageMatcher,
    NewColumnsEmptyMatcher,
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
import torch
from exasol_udf_mock_python.connection import Connection
from transformers import AutoTokenizer

from exasol_transformers_extension.udfs.models.ai_translate_extended_udf import AiTranslateExtendedUDF
from exasol_transformers_extension.utils.bucketfs_model_specification import (
    get_BucketFSModelSpecification_from_model_Specs,
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
def test_ai_translate_extended_udf(
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
        "max_new_tokens",
    ]

    sample_df = pd.DataFrame(data=sample_data, columns=columns)
    ctx = MockContext(input_df=sample_df)
    exa = MockExaEnvironment({bucketfs_conn_name: bucketfs_connection})

    sequence_classifier = AiTranslateExtendedUDF(exa, batch_size=batch_size)
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
    "description,  device_id, languages, max_new_tokens",
    [
        (
            "on CPU with max_new_tokens > expected result tokens",
            None,
            [("English", "French")],
            20,
        ),
        (
            "on CPU with max_new_tokens < expected result tokens",
            None,
            [("English", "French")],
            2,
        ),
        (
            "on GPU with max_new_tokens > expected result tokens",
            0,
            [("English", "French")],
            20,
        ),
        (
            "on GPU with max_new_tokens < expected result tokens",
            0,
            [("English", "French")],
            2,
        ),
    ],
)
def test_translation_udf_max_new_tokens_effective(
    description,
    device_id,
    languages,
    max_new_tokens,
    prepare_translation_model_for_local_bucketfs,
):
    if device_id is not None and not torch.cuda.is_available():
        pytest.skip(
            f"There is no available device({device_id}) " f"to execute the test"
        )

    bucketfs_base_path = prepare_translation_model_for_local_bucketfs
    bucketfs_conn_name = "bucketfs_connection"
    bucketfs_connection = create_mounted_bucketfs_connection(bucketfs_base_path)

    input_text = model_params.text_data

    # we load the test models tokenizer to convert output to tokens,
    # in order to check if max_new_tokens is respected in the output.
    model_specification = model_params.seq2seq_model_specs
    current_model_specs = get_BucketFSModelSpecification_from_model_Specs(
        model_specification, bucketfs_conn_name, model_params.sub_dir
    )
    model_path_in_bucketfs = current_model_specs.get_bucketfs_model_save_path()
    tokenizer = AutoTokenizer.from_pretrained(
        str(bucketfs_base_path / model_path_in_bucketfs)
    )

    batch_size = 2
    sample_data = [
        (
            None,
            bucketfs_conn_name,
            model_params.sub_dir,
            model_params.seq2seq_model_specs.model_name,
            input_text,
            src_lang,
            target_lang,
            max_new_tokens,
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
        "max_new_tokens",
    ]

    sample_df = pd.DataFrame(data=sample_data, columns=columns)
    ctx = MockContext(input_df=sample_df)
    exa = MockExaEnvironment({bucketfs_conn_name: bucketfs_connection})

    sequence_classifier = AiTranslateExtendedUDF(exa, batch_size=batch_size)
    sequence_classifier.run(ctx)

    result_df = ctx.get_emitted()[0][0]
    new_columns = ["translation_text", "error_message"]

    result = Result(result_df)
    print(result)
    assert (
        result
        == ShapeMatcher(columns=columns, new_columns=new_columns, n_rows=len(languages))
        and result == NoErrorMessageMatcher()
    )

    for translated_text in result_df["translation_text"]:
        translated_text_tokenized = tokenizer(
            translated_text, return_tensors="pt", return_attention_mask=False
        )
        translated_text_token_ids = translated_text_tokenized["input_ids"][0]
        # there is  an "end token" in the generated sequences witch does not
        # count toward max_new_tokens
        assert len(translated_text_token_ids) - 1 <= max_new_tokens


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
        "max_new_tokens",
    ]

    sample_df = pd.DataFrame(data=sample_data, columns=columns)
    ctx = MockContext(input_df=sample_df)
    exa = MockExaEnvironment({bucketfs_conn_name: bucketfs_connection})

    sequence_classifier = AiTranslateExtendedUDF(exa, batch_size=batch_size)
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
