from unittest.mock import Mock

import pandas as pd
import pytest

from exasol_transformers_extension.udfs.models.transformation.strip_whitespace_from_columns import (
    StripWhitespaceTransformation,
)
from exasol_transformers_extension.udfs.models.transformation.transformation_pipeline import (
    TransformationPipeline,
)

data = {
    "col1": [420, " test test ", " test "],
    "col2": ["test\n", 40, "45"],
    "col3": ["test\n", 40, " 45 "],
}
stripped_data = {
    "col1": [None, "test test", "test"],
    "col2": ["test", None, "45"],
    "col3": ["test\n", 40, " 45 "],
}


@pytest.mark.parametrize(
    "description, in_dataframe, cols_to_strip, expected_dataframe_shape, "
    "expected_error_message, expected_out_dataframe",
    [
        (
            "strip whitespace",
            pd.DataFrame(data),
            ["col1", "col2"],
            (3, 4),
            "None",
            pd.DataFrame(stripped_data),
        ),
        (
            "given col not exist",
            pd.DataFrame(data),
            ["col-non-exist", "col2"],
            (3, 4),
            "Traceback",
            pd.DataFrame(data),
        ),
        ("no cols given", pd.DataFrame(data), [], (3, 4), "None", pd.DataFrame(data)),
    ],
)
def test_strip_whitespace_transformation(
    description,
    in_dataframe,
    cols_to_strip,
    expected_dataframe_shape,
    expected_error_message,
    expected_out_dataframe,
):
    model_loader_mock = Mock()

    transformations = TransformationPipeline(
        [
            StripWhitespaceTransformation(
                expected_input_columns=cols_to_strip,
            ),
        ]
    )

    output_generator = transformations.execute(in_dataframe, model_loader_mock)

    for output_df in output_generator:
        with pd.option_context(
            "display.max_rows", None, "display.max_columns", None, "display.width", None
        ):
            print(output_df)
            print(output_df["error_message"][0])
        assert all(
            expected_error_message in str(error_message)
            for error_message in output_df["error_message"]
        )

        assert output_df.shape == expected_dataframe_shape

        output_df_drop_error = output_df.drop(columns=["error_message"])
        assert output_df_drop_error.equals(expected_out_dataframe)
