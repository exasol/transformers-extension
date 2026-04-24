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


@pytest.mark.parametrize(
    "description, in_dataframe, default_cols, additional_dict, expected_dataframe_shape, expected_error_message",
    [
        (
            "add all",
            pd.DataFrame(data),
            [
                "sub_dir",
                "bucketfs_conn",
                "device_id",
                "top_k",
                "return_ranks",
                "max_new_tokens",
                "return_full_text",
                "aggregation_strategy",
            ],
            None,
            (3, 11),
            "None",
        ),
        (
            "add none",
            pd.DataFrame(data),
            [],
            None,
            (3, 3),
            "None",
        ),
        (
            "add some",
            pd.DataFrame(data),
            [
                "sub_dir",
                "bucketfs_conn",
                "device_id",
                "top_k",
                "return_ranks",
            ],
            None,
            (3, 8),
            "None",
        ),
        (
            "add to empty df",
            pd.DataFrame(),
            [
                "sub_dir",
                "bucketfs_conn",
                "device_id",
                "top_k",
                "return_ranks",
                "max_new_tokens",
                "return_full_text",
                "aggregation_strategy",
            ],
            None,
            (
                0,
                9,
            ),  # the dataframe columns will be created, but no rows exist so no rows will be filled
            "None",
        ),
        (
            "add existing column",
            pd.DataFrame(data | _sub_dir),
            ["sub_dir"],
            None,
            (3, 4),
            "None",
        ),
        (
            "add additional default value",
            pd.DataFrame(data),
            ["sub_dir", "new_default_col", "model_name"],
            {
                "new_default_col": "new_default_value",
                "model_name": DEFAULT_MODEL_SPECS[
                    "AiSentimentUDF"
                ].model_name,
            },
            (3, 6),
            "None",
        ),
        (
            "overwrite existing default value",
            pd.DataFrame(data),
            ["sub_dir"],
            {"sub_dir": "another_sub_dir"},
            (3, 4),
            "None",
        ),
    ],
)
def test_add_default_columns_transformation(
    description,
    in_dataframe,
    default_cols,
    additional_dict,
    expected_dataframe_shape,
    expected_error_message,
):
    model_loader_mock = Mock()

    transformations = TransformationPipeline(
        [
            AddDefaultColumnsTransformation(
                new_columns=default_cols,
                default_values=additional_dict,
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
        if additional_dict is not None:
            assert all(
                def_col in output_df.columns for def_col in additional_dict.keys()
            )
        if "model_name" in default_cols:
            assert all(
                output_df["model_name"][i]
                == DEFAULT_MODEL_SPECS["AiSentimentUDF"].model_name
                for i in output_df.index
            )
            default_cols.remove("model_name")

        all_def_values = (
            DEFAULT_VALUES | additional_dict if additional_dict else DEFAULT_VALUES
        )
        assert all(
            output_df[def_col][i] == all_def_values[def_col]
            for def_col in default_cols
            for i in output_df.index
        )
        assert output_df.shape == expected_dataframe_shape


@pytest.mark.parametrize(
    "description, in_dataframe, default_cols, additional_dict, expected_dataframe_shape, expected_error_message",
    [
        (
            "add unknown column",
            pd.DataFrame(data),
            ["unknown"],
            None,
            (3, 4),  # the ensure_output_format still adds the column
            "KeyError: 'unknown'",
        ),
    ],
)
def test_add_wrong_default_columns_transformation(
    description,
    in_dataframe,
    default_cols,
    additional_dict,
    expected_dataframe_shape,
    expected_error_message,
):
    model_loader_mock = Mock()

    transformations = TransformationPipeline(
        [
            AddDefaultColumnsTransformation(
                new_columns=default_cols,
                default_values=additional_dict,
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
