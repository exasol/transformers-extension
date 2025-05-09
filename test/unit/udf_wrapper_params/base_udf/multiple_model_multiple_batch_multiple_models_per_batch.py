import dataclasses
from pathlib import PurePosixPath
from test.unit.udf_wrapper_params.base_udf.make_data_row_functions import (
    answer,
    bucketfs_conn,
    input_data,
    make_input_row,
    make_input_row_with_span,
    make_model_output_for_one_input_row,
    make_number_of_strings,
    make_output_row,
    make_output_row_with_span,
    model_name,
    score,
    sub_dir,
)

from exasol_udf_mock_python.connection import Connection


@dataclasses.dataclass
class MultipleModelMultipleBatchMultipleModelsPerBatch:
    """
    Multiple models, multiple batches, multiple models per batch
    """

    expected_model_counter = 4
    batch_size = 2
    data_size = 1

    bfs_conn1, bfs_conn2, bfs_conn3, bfs_conn4 = make_number_of_strings(
        bucketfs_conn, 4
    )
    subdir1, subdir2, subdir3, subdir4 = make_number_of_strings(sub_dir, 4)
    model_name1, model_name2, model_name3, model_name4 = make_number_of_strings(
        model_name, 4
    )
    input_data1, input_data2, input_data3, input_data4 = make_number_of_strings(
        input_data, 4
    )
    answer1, answer2, answer3, answer4 = make_number_of_strings(answer, 4)

    input_data = (
        make_input_row(
            bucketfs_conn=bfs_conn1,
            sub_dir=subdir1,
            model_name=model_name1,
            input_data=input_data1,
        )
        * data_size
        + make_input_row(
            bucketfs_conn=bfs_conn2,
            sub_dir=subdir2,
            model_name=model_name2,
            input_data=input_data2,
        )
        * data_size
        + make_input_row(
            bucketfs_conn=bfs_conn3,
            sub_dir=subdir3,
            model_name=model_name3,
            input_data=input_data3,
        )
        * data_size
        + make_input_row(
            bucketfs_conn=bfs_conn4,
            sub_dir=subdir4,
            model_name=model_name4,
            input_data=input_data4,
        )
        * data_size
    )
    output_data = (
        make_output_row(
            bucketfs_conn=bfs_conn1,
            sub_dir=subdir1,
            model_name=model_name1,
            input_data=input_data1,
            answer=answer1,
            score=score,
        )
        * data_size
        + make_output_row(
            bucketfs_conn=bfs_conn2,
            sub_dir=subdir2,
            model_name=model_name2,
            input_data=input_data2,
            answer=answer2,
            score=score + 0.1,
        )
        * data_size
        + make_output_row(
            bucketfs_conn=bfs_conn3,
            sub_dir=subdir3,
            model_name=model_name3,
            input_data=input_data3,
            answer=answer3,
            score=score + 0.2,
        )
        * data_size
        + make_output_row(
            bucketfs_conn=bfs_conn4,
            sub_dir=subdir4,
            model_name=model_name4,
            input_data=input_data4,
            answer=answer4,
            score=score + 0.3,
        )
        * data_size
    )

    work_with_span_input_data = (
        make_input_row_with_span(
            bucketfs_conn=bfs_conn1,
            sub_dir=subdir1,
            model_name=model_name1,
            input_data=input_data1,
        )
        * data_size
        + make_input_row_with_span(
            bucketfs_conn=bfs_conn2,
            sub_dir=subdir2,
            model_name=model_name2,
            input_data=input_data2,
        )
        * data_size
        + make_input_row_with_span(
            bucketfs_conn=bfs_conn3,
            sub_dir=subdir3,
            model_name=model_name3,
            input_data=input_data3,
        )
        * data_size
        + make_input_row_with_span(
            bucketfs_conn=bfs_conn4,
            sub_dir=subdir4,
            model_name=model_name4,
            input_data=input_data4,
        )
        * data_size
    )
    work_with_span_output_data = (
        make_output_row_with_span(
            bucketfs_conn=bfs_conn1,
            sub_dir=subdir1,
            model_name=model_name1,
            input_data=input_data1,
            answer=answer1,
            score=score,
        )
        * data_size
        + make_output_row_with_span(
            bucketfs_conn=bfs_conn2,
            sub_dir=subdir2,
            model_name=model_name2,
            input_data=input_data2,
            answer=answer2,
            score=score + 0.1,
        )
        * data_size
        + make_output_row_with_span(
            bucketfs_conn=bfs_conn3,
            sub_dir=subdir3,
            model_name=model_name3,
            input_data=input_data3,
            answer=answer3,
            score=score + 0.2,
        )
        * data_size
        + make_output_row_with_span(
            bucketfs_conn=bfs_conn4,
            sub_dir=subdir4,
            model_name=model_name4,
            input_data=input_data4,
            answer=answer4,
            score=score + 0.3,
        )
        * data_size
    )

    tokenizer_model_output_df_model1 = [
        make_model_output_for_one_input_row(answer=answer1, score=score) * data_size
    ]
    tokenizer_model_output_df_model2 = [
        make_model_output_for_one_input_row(answer=answer2, score=score + 0.1)
        * data_size
    ]
    tokenizer_model_output_df_model3 = [
        make_model_output_for_one_input_row(answer=answer3, score=score + 0.2)
        * data_size
    ]
    tokenizer_model_output_df_model4 = [
        make_model_output_for_one_input_row(answer=answer4, score=score + 0.3)
        * data_size
    ]

    tokenizer_models_output_df = [
        tokenizer_model_output_df_model1,
        tokenizer_model_output_df_model2,
        tokenizer_model_output_df_model3,
        tokenizer_model_output_df_model4,
    ]

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, bfs_conn1)
    base_cache_dir2 = PurePosixPath(tmpdir_name, bfs_conn2)
    base_cache_dir3 = PurePosixPath(tmpdir_name, bfs_conn3)
    base_cache_dir4 = PurePosixPath(tmpdir_name, bfs_conn4)
    bfs_connections = {
        bfs_conn1: Connection(address=f"file://{base_cache_dir1}"),
        bfs_conn2: Connection(address=f"file://{base_cache_dir2}"),
        bfs_conn3: Connection(address=f"file://{base_cache_dir3}"),
        bfs_conn4: Connection(address=f"file://{base_cache_dir4}"),
    }
