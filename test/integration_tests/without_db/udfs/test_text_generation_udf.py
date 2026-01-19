from test.integration_tests.without_db.udfs.utils.matcher import (
    ColumnsMatcher,
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

from exasol_transformers_extension.udfs.models.text_generation_udf import (
    TextGenerationUDF,
)
from exasol_transformers_extension.utils.bucketfs_model_specification import (
    get_BucketFSModelSpecification_from_model_Specs,
)


@pytest.mark.parametrize(
    "description,  device_id, n_rows",
    [
        ("on CPU with batch input", None, 3),
        ("on CPU with single input", None, 1),
        ("on GPU with batch input", 0, 3),
        ("on GPU with single input", 0, 1),
    ],
)
def test_text_generation_udf(
    description, device_id, n_rows, prepare_text_generation_model_for_local_bucketfs
):
    if device_id is not None and not torch.cuda.is_available():
        pytest.skip(
            f"There is no available device({device_id}) " f"to execute the test"
        )

    bucketfs_base_path = prepare_text_generation_model_for_local_bucketfs
    bucketfs_conn_name = "bucketfs_connection"
    bucketfs_connection = create_mounted_bucketfs_connection(bucketfs_base_path)

    batch_size = 2
    text_data = "Exasol is an analytics database management"
    max_new_tokens = 10
    return_full_text = True
    sample_data = [
        (
            None,
            bucketfs_conn_name,
            model_params.sub_dir,
            model_params.text_gen_model_specs.model_name,
            text_data,
            max_new_tokens,
            return_full_text,
        )
        for _ in range(n_rows)
    ]
    columns = [
        "device_id",
        "bucketfs_conn",
        "sub_dir",
        "model_name",
        "text_data",
        "max_new_tokens",
        "return_full_text",
    ]

    sample_df = pd.DataFrame(data=sample_data, columns=columns)
    ctx = MockContext(input_df=sample_df)
    exa = MockExaEnvironment({bucketfs_conn_name: bucketfs_connection})

    sequence_classifier = TextGenerationUDF(exa, batch_size=batch_size)
    sequence_classifier.run(ctx)

    result_df = ctx.get_emitted()[0][0]
    new_columns = ["generated_text", "error_message"]

    result = Result(result_df)
    assert (
        result == ShapeMatcher(columns=columns, new_columns=new_columns, n_rows=n_rows)
        and result == ColumnsMatcher(columns=columns[1:], new_columns=new_columns)
        and result == NoErrorMessageMatcher()
        and result_df["generated_text"].str.contains(text_data).all()
    )


@pytest.mark.parametrize(
    "description,  device_id, max_new_tokens",
    [
        ("on CPU with max_new_tokens > expected result tokens", None, 20),
        ("on CPU with max_new_tokens < expected result tokens", None, 2),
        ("on GPU with max_new_tokens > expected result tokens", 0, 20),
        ("on GPU with max_new_tokens < expected result tokens", 0, 2),
    ],
)
def test_text_generation_udf(
    description,
    device_id,
    max_new_tokens,
    prepare_text_generation_model_for_local_bucketfs,
):
    if device_id is not None and not torch.cuda.is_available():
        pytest.skip(
            f"There is no available device({device_id}) " f"to execute the test"
        )

    bucketfs_base_path = prepare_text_generation_model_for_local_bucketfs
    bucketfs_conn_name = "bucketfs_connection"
    bucketfs_connection = create_mounted_bucketfs_connection(bucketfs_base_path)

    batch_size = 2
    text_data = "Exasol is an analytics database management test test test"
    n_input_tokens = len(text_data.split())
    # we load the test models tokenizer to convert input and output to tokens,
    # in order to check if max_new_tokens is respected in the output.
    model_specification = model_params.text_gen_model_specs
    current_model_specs = get_BucketFSModelSpecification_from_model_Specs(
        model_specification, bucketfs_conn_name, model_params.sub_dir
    )
    model_path_in_bucketfs = current_model_specs.get_bucketfs_model_save_path()
    tokenizer = AutoTokenizer.from_pretrained(
        str(bucketfs_base_path / model_path_in_bucketfs)
    )

    input_tokenized = tokenizer(
        text_data, return_tensors="pt", return_attention_mask=False
    )
    input_token_ids = input_tokenized["input_ids"][0]

    return_full_text = True
    sample_data = [
        (
            None,
            bucketfs_conn_name,
            model_params.sub_dir,
            model_params.text_gen_model_specs.model_name,
            text_data,
            max_new_tokens,
            return_full_text,
        )
    ]
    columns = [
        "device_id",
        "bucketfs_conn",
        "sub_dir",
        "model_name",
        "text_data",
        "max_new_tokens",
        "return_full_text",
    ]

    sample_df = pd.DataFrame(data=sample_data, columns=columns)
    ctx = MockContext(input_df=sample_df)
    exa = MockExaEnvironment({bucketfs_conn_name: bucketfs_connection})

    sequence_classifier = TextGenerationUDF(exa, batch_size=batch_size)
    sequence_classifier.run(ctx)

    result_df = ctx.get_emitted()[0][0]
    new_columns = ["generated_text", "error_message"]

    result = Result(result_df)

    assert (
        result == ShapeMatcher(columns=columns, new_columns=new_columns, n_rows=1)
        and result == ColumnsMatcher(columns=columns[1:], new_columns=new_columns)
        and result == NoErrorMessageMatcher()
        and result_df["generated_text"].str.contains(text_data).all()
    )

    for generated_text in result_df["generated_text"]:
        generated_text_tokenized = tokenizer(
            generated_text, return_tensors="pt", return_attention_mask=False
        )
        generated_text_token_ids = tokenizer.convert_ids_to_tokens(
            generated_text_tokenized["input_ids"][0]
        )
        assert len(generated_text_token_ids) - len(input_token_ids) <= max_new_tokens


@pytest.mark.parametrize(
    "description,  device_id, n_rows",
    [
        ("on CPU with batch input", None, 3),
        ("on CPU with single input", None, 1),
        ("on GPU with batch input", 0, 3),
        ("on GPU with single input", 0, 1),
    ],
)
def test_text_generation_udf_on_error_handlig(
    description, device_id, n_rows, prepare_text_generation_model_for_local_bucketfs
):
    if device_id is not None and not torch.cuda.is_available():
        pytest.skip(
            f"There is no available device({device_id}) " f"to execute the test"
        )

    bucketfs_base_path = prepare_text_generation_model_for_local_bucketfs
    bucketfs_conn_name = "bucketfs_connection"
    bucketfs_connection = Connection(address=f"file://{bucketfs_base_path}")

    batch_size = 2
    text_data = "Exasol is an analytics database management"
    max_new_tokens = 10
    return_full_text = True
    sample_data = [
        (
            None,
            bucketfs_conn_name,
            model_params.sub_dir,
            "not existing model",
            text_data,
            max_new_tokens,
            return_full_text,
        )
        for _ in range(n_rows)
    ]
    columns = [
        "device_id",
        "bucketfs_conn",
        "sub_dir",
        "model_name",
        "text_data",
        "max_new_tokens",
        "return_full_text",
    ]

    sample_df = pd.DataFrame(data=sample_data, columns=columns)
    ctx = MockContext(input_df=sample_df)
    exa = MockExaEnvironment({bucketfs_conn_name: bucketfs_connection})

    sequence_classifier = TextGenerationUDF(exa, batch_size=batch_size)
    sequence_classifier.run(ctx)

    result_df = ctx.get_emitted()[0][0]
    new_columns = ["generated_text", "error_message"]

    result = Result(result_df)
    assert (
        result == ShapeMatcher(columns=columns, new_columns=new_columns, n_rows=n_rows)
        and result == NewColumnsEmptyMatcher(new_columns=new_columns)
        and result == ErrorMessageMatcher()
    )
