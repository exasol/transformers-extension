import dataclasses
from pathlib import PurePosixPath

from exasol_transformers_extension.deployment.default_udf_parameters import DEFAULT_BUCKETFS_CONN_NAME, DEFAULT_VALUES
from test.unit.udf_wrapper_params.ai_translate_extended.make_data_row_functions import (
    target_language,
    translation_text, text_data,
    source_language, error_msg,
    make_translation_text
)

from exasol_udf_mock_python.connection import Connection


@dataclasses.dataclass
class DefaultValuesMultipleBatchComplete:

    expected_model_counter = 1
    batch_size = 2
    data_size = 2

    target_language_1 = target_language
    target_language_2 = "French"

    translation_text_1 = translation_text
    translation_text_2 = "text traduit, "

    input_data = (
        [[
            text_data,
            source_language,
            target_language_1
        ]] * data_size +
        [[
            text_data,
            source_language,
            target_language_2
        ]] * data_size
    )

    output_data = (
            [
                (
                    text_data,
                    source_language,
                    target_language_1,
                    make_translation_text(translation_text_1,
                                          max_new_tokens=DEFAULT_VALUES["max_new_tokens"],
                                          error_msg=error_msg),
                    error_msg,
                )
            ] * data_size +
            [
                (
                    text_data,
                    source_language,
                    target_language_2,
                    make_translation_text(translation_text_2,
                                          max_new_tokens=DEFAULT_VALUES["max_new_tokens"],
                                          error_msg=error_msg),
                    error_msg,
                )
            ] * data_size

    )

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir = PurePosixPath(tmpdir_name, DEFAULT_BUCKETFS_CONN_NAME)
    bfs_connections = {
        DEFAULT_BUCKETFS_CONN_NAME: Connection(address=f"file://{base_cache_dir}"),
    }
