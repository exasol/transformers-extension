from pathlib import PurePosixPath
import dataclasses

from exasol_udf_mock_python.connection import Connection

from test.unit.udf_wrapper_params.translation.make_data_row_functions import (
    bucketfs_conn,
    make_input_row,
    make_model_output_for_one_input_row,
    make_udf_output_for_one_input_row, sub_dir, model_name, text_data, translation_text,
)
from test.unit.utils.utils_for_udf_tests import make_number_of_strings


@dataclasses.dataclass
class ErrorOnPredictionMultipleModelMultipleBatch:
    """
    not cached error, multiple model, multiple batch
    """

    expected_model_counter = 2
    batch_size = 3
    data_size = 2

    bfs_conn1, bfs_conn2 = make_number_of_strings(bucketfs_conn,2)
    sub_dir1, sub_dir2 = make_number_of_strings(sub_dir,2)
    model1, model2 = make_number_of_strings(model_name,2)

    input_data = (make_input_row(bucketfs_conn=bfs_conn1,
                                 sub_dir=sub_dir1,
                                 model_name=model1,
                                 text_data=text_data) * data_size +
                  make_input_row(bucketfs_conn=bfs_conn2,
                                 sub_dir=sub_dir2,
                                 model_name=model2,
                                 text_data= "error on pred") * data_size)

    output_data = (make_udf_output_for_one_input_row(bucketfs_conn=bfs_conn1,
                                                     sub_dir=sub_dir1,
                                                     model_name=model1,
                                                     text_data=text_data,
                                                     translation_text=translation_text) * data_size +
                   make_udf_output_for_one_input_row(bucketfs_conn=bfs_conn2,
                                                     sub_dir=sub_dir2,
                                                     model_name=model2,
                                                     text_data= "error on pred",
                                                     translation_text=None,
                                                     error_msg="Traceback") * data_size)

    translation_models_output_df = [
        [(make_model_output_for_one_input_row(translation_text) * data_size)] +
        [([Exception("Traceback mock_pipeline is " "throwing an error intentionally")])],
        [([Exception("Traceback mock_pipeline is " "throwing an error intentionally")])]#todo error in model output?
    ]

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, bfs_conn1)
    base_cache_dir2 = PurePosixPath(tmpdir_name, bfs_conn2)
    bfs_connections = {
        bfs_conn1: Connection(address=f"file://{base_cache_dir1}"),
        bfs_conn2: Connection(address=f"file://{base_cache_dir2}"),
    }


