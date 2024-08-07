from tests.integration_tests.with_db.udfs.python_rows_to_sql import python_rows_to_sql
from tests.utils.parameters import model_params


def test_zero_shot_classification_single_text_script(
        setup_database, db_conn, upload_zero_shot_classification_model_to_bucketfs):
    bucketfs_conn_name, _ = setup_database
    n_rows = 100
    input_data = []
    candidate_labels = "Database,Analytics,Germany,Food,Party"
    n_labels = len(candidate_labels.split(","))
    for i in range(n_rows):
        input_data.append((
            '',
            bucketfs_conn_name,
            str(model_params.sub_dir),
            model_params.zero_shot_model_specs.model_name,
            'The database software company Exasol is based in Nuremberg',
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
    added_columns = 4  # label,score,rank,error_message
    removed_columns = 1  # device_id
    n_rows_result = n_rows * n_labels
    n_cols_result = len(input_data[0]) + (added_columns - removed_columns)
    assert len(result) == n_rows_result and len(result[0]) == n_cols_result

    # lenient test for quality of results, will be replaced by deterministic test later
    acceptable_results = ["Analytics", "Database", "Germany"]

    def contains(string, list):
        return any(map(lambda x: x in string, list))

    number_accepted_results = 0
    for i in range(len(result)):
        if (contains(result[i][5], acceptable_results) and
                result[i][6] > 0.8):  # check if confidence resonably high
            number_accepted_results += 1
        elif result[i][6] < 0.2:
            number_accepted_results += 1
    assert number_accepted_results > n_rows_result / 1.5
