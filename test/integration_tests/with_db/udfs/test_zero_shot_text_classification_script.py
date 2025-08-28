from test.integration_tests.utils.model_output_quality_checkers import (
    assert_lenient_check_of_output_quality_with_score,
)
from test.integration_tests.utils.model_output_result_number_checker import (
    assert_correct_number_of_results,
)
from test.integration_tests.with_db.udfs.python_rows_to_sql import python_rows_to_sql
from test.utils.parameters import model_params


def setup_common_input_data():
    n_rows = 100
    candidate_labels = "Database,Analytics,Germany,Food,Party"
    n_labels = len(candidate_labels.split(","))
    n_rows_result = n_rows * n_labels
    text_data = "The database software company Exasol is based in Nuremberg"
    return n_rows, candidate_labels, n_labels, n_rows_result, text_data


def test_zero_shot_classification_single_text_script_without_spans(
    setup_database, db_conn, upload_zero_shot_classification_model_to_bucketfs
):
    bucketfs_conn_name, _ = setup_database
    n_rows, candidate_labels, n_labels, n_rows_result, text_data = (
        setup_common_input_data()
    )
    input_data = []

    for i in range(n_rows):
        input_data.append(
            (
                "",
                bucketfs_conn_name,
                str(model_params.sub_dir),
                model_params.zero_shot_model_specs.model_name,
                text_data,
                candidate_labels,
            )
        )

    query = (
        f"SELECT TE_ZERO_SHOT_TEXT_CLASSIFICATION_UDF("
        f"t.device_id, "
        f"t.bucketfs_conn_name, "
        f"t.sub_dir, "
        f"t.model_name, "
        f"t.text_data,"
        f"t.candidate_labels) "
        f"FROM (VALUES {python_rows_to_sql(input_data)} "
        f"AS t(device_id, bucketfs_conn_name, "
        f"sub_dir, model_name, text_data, candidate_labels));"
    )

    # execute sequence classification UDF
    result = db_conn.execute(query).fetchall()

    # assertions
    assert result[0][-1] is None
    # added_columns: label,score,rank,error_message
    # removed_columns: device_id
    assert_correct_number_of_results(4, 1, input_data[0], result, n_rows_result)

    acceptable_results = ["Analytics", "Database", "Germany"]
    assert_lenient_check_of_output_quality_with_score(
        result, acceptable_results, 1/1.8
    )


def test_zero_shot_classification_single_text_script_with_spans(
    setup_database, db_conn, upload_zero_shot_classification_model_to_bucketfs
):
    bucketfs_conn_name, _ = setup_database
    n_rows, candidate_labels, n_labels, n_rows_result, text_data = (
        setup_common_input_data()
    )
    input_data = []

    for i in range(n_rows):
        input_data.append(
            (
                "",
                bucketfs_conn_name,
                str(model_params.sub_dir),
                model_params.zero_shot_model_specs.model_name,
                text_data,
                i,
                0,
                len(text_data),
                candidate_labels,
            )
        )

    query = (
        f"SELECT TE_ZERO_SHOT_TEXT_CLASSIFICATION_UDF_WITH_SPAN("
        f"t.device_id, "
        f"t.bucketfs_conn_name, "
        f"t.sub_dir, "
        f"t.model_name, "
        f"t.text_data,"
        f"t.text_data_doc_id, "
        f"t.text_data_char_begin, "
        f"t.text_data_char_end, "
        f"t.candidate_labels) "
        f"FROM (VALUES {python_rows_to_sql(input_data)} "
        f"AS t(device_id, bucketfs_conn_name, "
        f"sub_dir, model_name, text_data, text_data_doc_id, text_data_char_begin, "
        f"text_data_char_end, candidate_labels));"
    )

    # execute sequence classification UDF
    result = db_conn.execute(query).fetchall()

    # assertions
    assert result[0][-1] is None
    # added_columns: label,score,rank,error_message
    # removed_columns: device_id, text_data, candidate_labels
    assert_correct_number_of_results(4, 3, input_data[0], result, n_rows_result)

    acceptable_results = ["Analytics", "Database", "Germany"]
    assert_lenient_check_of_output_quality_with_score(
        result, acceptable_results, 1/1.8, label_index=6
    )
