from tests.integration_tests.with_db.udfs.python_rows_to_sql import python_rows_to_sql
from tests.utils.parameters import model_params

def setup_common_input_data():
    n_rows = 100
    candidate_labels = "Database,Analytics,Germany,Food,Party"
    n_labels = len(candidate_labels.split(","))
    n_rows_result = n_rows * n_labels
    text_data = 'The database software company Exasol is based in Nuremberg'
    return n_rows, candidate_labels, n_labels, n_rows_result, text_data

def assert_correct_number_of_results(added_columns: int, removed_columns: int,
                              input_data_row: tuple, result: list, n_rows_result: int):
    n_cols_result = len(input_data_row) + (added_columns - removed_columns)
    assert len(result) == n_rows_result and len(result[0]) == n_cols_result, (f"format of result is not correct,"
                                                                              f"expected {n_rows_result} rows, {n_cols_result} columns."
                                                                              f"actual: {len(result)} rows, {len(result[0])} columns")

def assert_lenient_check_of_output_quality(result: list, n_rows_result: int, label_index: int = 5):
    acceptable_results = ["Analytics", "Database", "Germany"]

    def contains(string, list):
        return any(map(lambda x: x in string, list))

    number_accepted_results = 0
    for i in range(len(result)):
        if (contains(result[i][label_index], acceptable_results) and
                result[i][label_index+1] > 0.8):  # check if confidence reasonably high
            number_accepted_results += 1
        elif result[i][label_index+1] < 0.2:
            number_accepted_results += 1
    assert number_accepted_results > n_rows_result / 1.5, f"Not enough acceptable labels ({acceptable_results}) in results {result}"

def test_zero_shot_classification_single_text_script_without_spans(
        setup_database, db_conn, upload_zero_shot_classification_model_to_bucketfs):
    bucketfs_conn_name, _ = setup_database
    n_rows, candidate_labels, n_labels, n_rows_result, text_data = (
        setup_common_input_data())
    input_data = []

    for i in range(n_rows):
        input_data.append((
            '',
            bucketfs_conn_name,
            str(model_params.sub_dir),
            model_params.zero_shot_model_specs.model_name,
            text_data,
            candidate_labels
        ))

    query = f"SELECT TE_ZERO_SHOT_TEXT_CLASSIFICATION_UDF(" \
            f"t.device_id, " \
            f"t.bucketfs_conn_name, " \
            f"t.sub_dir, " \
            f"t.model_name, " \
            f"t.text_data," \
            f"t.candidate_labels) " \
            f"FROM (VALUES {python_rows_to_sql(input_data)} " \
            f"AS t(device_id, bucketfs_conn_name, " \
            f"sub_dir, model_name, text_data, candidate_labels));"

    # execute sequence classification UDF
    result = db_conn.execute(query).fetchall()

    # assertions
    assert result[0][-1] is None
    # added_columns: label,score,rank,error_message
    # removed_columns: device_id
    assert_correct_number_of_results(4, 1,input_data[0], result, n_rows_result)

    # lenient test for quality of results, will be replaced by deterministic test later
    assert_lenient_check_of_output_quality(result, n_rows_result)

def test_zero_shot_classification_single_text_script_with_spans(
        setup_database, db_conn, upload_zero_shot_classification_model_to_bucketfs):
    bucketfs_conn_name, _ = setup_database
    n_rows, candidate_labels, n_labels, n_rows_result, text_data = (
        setup_common_input_data())
    input_data = []

    for i in range(n_rows):
        input_data.append((
            '',
            bucketfs_conn_name,
            str(model_params.sub_dir),
            model_params.zero_shot_model_specs.model_name,
            text_data,
            i,
            0,
            len(text_data),
            candidate_labels
        ))

    query = f"SELECT TE_ZERO_SHOT_TEXT_CLASSIFICATION_UDF_WITH_SPAN(" \
            f"t.device_id, " \
            f"t.bucketfs_conn_name, " \
            f"t.sub_dir, " \
            f"t.model_name, " \
            f"t.text_data," \
            f"t.text_data_doc_id, " \
            f"t.text_data_char_begin, " \
            f"t.text_data_char_end, " \
            f"t.candidate_labels) " \
            f"FROM (VALUES {python_rows_to_sql(input_data)} " \
            f"AS t(device_id, bucketfs_conn_name, " \
            f"sub_dir, model_name, text_data, text_data_doc_id, text_data_char_begin, " \
            f"text_data_char_end, candidate_labels));"

    # execute sequence classification UDF
    result = db_conn.execute(query).fetchall()

    # assertions
    assert result[0][-1] is None
    # added_columns: label,score,rank,error_message
    # removed_columns: device_id, text_data, candidate_labels
    assert_correct_number_of_results(4, 3,input_data[0], result, n_rows_result)

    # lenient test for quality of results, will be replaced by deterministic test later
    assert_lenient_check_of_output_quality(result, n_rows_result, label_index=6)

