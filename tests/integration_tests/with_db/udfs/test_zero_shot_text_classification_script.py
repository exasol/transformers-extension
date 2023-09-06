from tests.integration_tests.with_db.udfs.python_rows_to_sql import python_rows_to_sql
from tests.utils.parameters import model_params


def test_sequence_classification_single_text_script(
        setup_database, pyexasol_connection, upload_base_model_to_bucketfs):
    bucketfs_conn_name, schema_name = setup_database
    n_rows = 100
    input_data = []
    candidate_labels = "Database,Analytics,Germany,Food,Party"
    n_labels = len(candidate_labels.split(","))
    for i in range(n_rows):
        input_data.append((
            '',
            bucketfs_conn_name,
            None,
            str(model_params.sub_dir),
            model_params.base_model,
            model_params.text_data,
            candidate_labels
        ))

    query = f"SELECT TE_ZERO_SHOT_TEXT_CLASSIFICATION_UDF(" \
            f"t.device_id, " \
            f"t.bucketfs_conn_name, " \
            f"t.token_conn_name, " \
            f"t.sub_dir, " \
            f"t.model_name, " \
            f"t.text_data," \
            f"t.candidate_labels) " \
            f"FROM (VALUES {python_rows_to_sql(input_data)} " \
            f"AS t(device_id, bucketfs_conn_name, token_conn_name, " \
            f"sub_dir, model_name, text_data, candidate_labels));"

    # execute sequence classification UDF
    result = pyexasol_connection.execute(query).fetchall()

    # assertions
    added_columns = 4  # label,score,rank,error_message
    removed_columns = 1  # device_id
    n_rows_result = n_rows * n_labels
    n_cols_result = len(input_data[0]) + (added_columns - removed_columns)
    assert len(result) == n_rows_result and len(result[0]) == n_cols_result
