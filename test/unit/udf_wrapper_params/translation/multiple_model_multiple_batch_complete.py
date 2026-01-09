from pathlib import PurePosixPath
import dataclasses

from exasol_udf_mock_python.connection import Connection
from test.unit.udf_wrapper_params.translation.make_data_row_functions import (
    bucketfs_conn,
    make_input_row,
    make_model_output_for_one_input_row,
    make_udf_output_for_one_input_row, sub_dir, model_name, text_data,
)
from test.unit.utils.utils_for_udf_tests import make_number_of_strings


@dataclasses.dataclass
class MultipleModelMultipleBatchComplete:
    """
    multiple model, multiple batch, last batch complete
    """

    expected_model_counter = 2
    batch_size = 2
    data_size = 2
    bfs_conn1, bfs_conn2 = make_number_of_strings(bucketfs_conn,2)
    sub_dir1, sub_dir2 = make_number_of_strings(sub_dir,2)
    model1, model2 = make_number_of_strings(model_name,2)
    text1, text2 = make_number_of_strings(text_data, 2)

    translation_text1 = text1 + "übersetzt"
    translation_text2 = text2 + "übersetzt"

    input_data = (make_input_row(bucketfs_conn=bfs_conn1,
                                 sub_dir=sub_dir1,
                                 model_name=model1,
                                 text_data=text1) * data_size +
                  make_input_row(bucketfs_conn=bfs_conn2,
                                 sub_dir=sub_dir2,
                                 model_name=model2,
                                 text_data=text2) * data_size)

    output_data = (make_udf_output_for_one_input_row(bucketfs_conn=bfs_conn1,
                                                     sub_dir=sub_dir1,
                                                     model_name=model1,
                                                     text_data=text1,
                                                     translation_text=translation_text1) * data_size +
                   make_udf_output_for_one_input_row(bucketfs_conn=bfs_conn2,
                                                     sub_dir=sub_dir2,
                                                     model_name=model2,
                                                     text_data=text2,
                                                     translation_text=translation_text2) * data_size)

    translation_model_output_df_batch1 = [
        make_model_output_for_one_input_row(translation_text1) * data_size
    ]
    translation_model_output_df_batch2 = [
        make_model_output_for_one_input_row(translation_text2) * data_size
    ]

    translation_models_output_df = [
        translation_model_output_df_batch1,
        translation_model_output_df_batch2,
    ]

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, bfs_conn1)
    base_cache_dir2 = PurePosixPath(tmpdir_name, bfs_conn2)
    bfs_connections = {
        bfs_conn1: Connection(address=f"file://{base_cache_dir1}"),
        bfs_conn2: Connection(address=f"file://{base_cache_dir2}"),
    }


