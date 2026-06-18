from unittest.mock import Mock

import pandas as pd
import pytest

from exasol_transformers_extension.udfs.models.transformation.remove_columns import (
    RemoveColumnsTransformation,
)
from exasol_transformers_extension.udfs.models.transformation.transformation_pipeline import (
    TransformationPipeline,
)

data = {"col1": [420, 380, 390], "col2": [50, 40, 45]}

_err = {"error_message": [None, None, None]}


@pytest.mark.parametrize(
    "description, in_dataframe, remove_cols, expected_dataframe, expected_error_message",
    [
        (
            "remove one",
            pd.DataFrame(data),
            ["col1"],
            pd.DataFrame({"col2": [50, 40, 45]}),
            "None",
        ),
        ("remove all", pd.DataFrame(data), ["col1", "col2"], pd.DataFrame(), "None"),
        ("remove all", pd.DataFrame(data), [], pd.DataFrame(data), "None"),
        (
            "remove non-existing",
            pd.DataFrame(data),
            ["col3"],
            pd.DataFrame(data),
            "ValueError: Missing expected input columns for RemoveColumnsTransformation",
        ),
    ],
)
def test_remove_columns_transformation(
    description, in_dataframe, remove_cols, expected_dataframe, expected_error_message
):
    model_loader_mock = Mock()

    transformations = TransformationPipeline(
        [
            RemoveColumnsTransformation(
                removed_columns=remove_cols,
            ),
        ]
    )

    output_generator = transformations.execute(in_dataframe, model_loader_mock)

    for output_df in output_generator:
        assert all(
            expected_error_message in str(error_message)
            for error_message in output_df["error_message"]
        )
        if expected_dataframe.empty:
            assert output_df.drop(
                columns=["error_message"]
            ).empty  # this on has an index, so not equal to empty df
        else:
            assert expected_dataframe.equals(output_df.drop(columns=["error_message"]))
