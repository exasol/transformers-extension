from unittest.mock import Mock

import pandas as pd
import pytest

from exasol_transformers_extension.deployment.default_udf_parameters import (
    DEFAULT_MODEL_SPECS,
    DEFAULT_VALUES,
)
from exasol_transformers_extension.udfs.models.transformation.add_default_columns import (
    AddDefaultColumnsTransformation,
)
from exasol_transformers_extension.udfs.models.transformation.transformation_pipeline import (
    TransformationPipeline,
)

data = {"col1": [420, 380, 390], "col2": [50, 40, 45]}

_err = {"error_message": [None, None, None]}
_sub_dir = {"sub_dir": [DEFAULT_VALUES["sub_dir"], "another_subdir", None]}


# todo we dont check existing cols dont get changed. do we need to?
@pytest.mark.parametrize(
    "description, in_dataframe, default_cols, expected_dataframe_shape, expected_error_message, udf_name",
    [
        (
            "add all",
            pd.DataFrame(data),
            [
                "model_name",
                "sub_dir",
                "bucketfs_conn",
                "device_id",
                "top_k",
                "return_ranks",
                "max_new_tokens",
                "return_full_text",
                "aggregation_strategy",
            ],
            (3, 12),
            "None",
            "model_for_a_specific_udf",
        ),
        (
            "add none",
            pd.DataFrame(data),
            [],
            (3, 3),
            "None",
            "model_for_a_specific_udf",
        ),
        (
            "add some",
            pd.DataFrame(data),
            [
                "model_name",
                "sub_dir",
                "bucketfs_conn",
                "device_id",
                "top_k",
                "return_ranks",
            ],
            (3, 9),
            "None",
            "model_for_a_specific_udf",
        ),
        (
            "add to empty df",
            pd.DataFrame(),
            [
                "model_name",
                "sub_dir",
                "bucketfs_conn",
                "device_id",
                "top_k",
                "return_ranks",
                "max_new_tokens",
                "return_full_text",
                "aggregation_strategy",
            ],
            (
                0,
                10,
            ),  # the dataframe columns will be created, but no rows exist so no rows will be filled
            "None",
            "model_for_a_specific_udf",
        ),
        (
            "add existing column",
            pd.DataFrame(data | _sub_dir),
            ["sub_dir"],
            (3, 4),
            "None",
            "model_for_a_specific_udf",
        ),
    ],
)
def test_add_default_columns_transformation(
    description,
    in_dataframe,
    default_cols,
    expected_dataframe_shape,
    expected_error_message,
    udf_name,
):
    model_loader_mock = Mock()

    transformations = TransformationPipeline(
        [
            AddDefaultColumnsTransformation(
                expected_input_columns=[],
                new_columns=default_cols,
                removed_columns=[],
                udf_name=udf_name,
            ),
        ]
    )

    output_generator = transformations.execute(in_dataframe, model_loader_mock)

    for output_df in output_generator:
        assert all(
            expected_error_message in str(error_message)
            for error_message in output_df["error_message"]
        )

        assert all(def_col in output_df.columns for def_col in default_cols)
        if "model_name" in default_cols:
            assert all(
                output_df["model_name"][i] == DEFAULT_MODEL_SPECS[udf_name].model_name
                for i in output_df.index
            )
            default_cols.remove("model_name")
        assert all(
            output_df[def_col][i] == DEFAULT_VALUES[def_col]
            for def_col in default_cols
            for i in output_df.index
        )
        assert output_df.shape == expected_dataframe_shape


@pytest.mark.parametrize(
    "description, in_dataframe, default_cols, expected_dataframe_shape, expected_error_message, udf_name",
    [
        (
            "add unknown column",
            pd.DataFrame(data),
            ["unknown"],
            (3, 4),  # the ensure_output_format still adds the column
            "KeyError: 'unknown'",
            "model_for_a_specific_udf",
        ),
        (
            "add unknown udf_model",
            pd.DataFrame(data),
            ["model_name"],
            (3, 4),  # the ensure_output_format still adds the column
            "KeyError: 'unknown_udf_name'",
            "unknown_udf_name",
        ),
    ],
)
def test_add_wrong_default_columns_transformation(
    description,
    in_dataframe,
    default_cols,
    expected_dataframe_shape,
    expected_error_message,
    udf_name,
):
    model_loader_mock = Mock()

    transformations = TransformationPipeline(
        [
            AddDefaultColumnsTransformation(
                expected_input_columns=[],
                new_columns=default_cols,
                removed_columns=[],
                udf_name=udf_name,
            ),
        ]
    )

    output_generator = transformations.execute(in_dataframe, model_loader_mock)

    for output_df in output_generator:
        assert all(
            expected_error_message in str(error_message)
            for error_message in output_df["error_message"]
        )

        assert all(def_col in output_df.columns for def_col in default_cols)
        assert all(
            output_df[def_col][i] is None
            for def_col in default_cols
            for i in output_df.index
        )
        assert output_df.shape == expected_dataframe_shape
