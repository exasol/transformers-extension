from tests.integration_tests.with_db.udfs.python_rows_to_sql import python_rows_to_sql
from tests.utils.parameters import model_params


def test_sequence_classification_text_pair_script(
        setup_database, pyexasol_connection, upload_base_model_to_bucketfs):
    bucketfs_conn_name, _ = setup_database
    n_labels = 2
    n_rows = 100
    input_data = []
    for i in range(n_rows):
        input_data.append((
            '',
            bucketfs_conn_name,
            str(model_params.sub_dir),
            model_params.base_model_specs.model_name,
            model_params.text_data,
            ' '.join((model_params.text_data, str(i)))))

    query = f"SELECT TE_SEQUENCE_CLASSIFICATION_TEXT_PAIR_UDF(" \
            f"t.device_id, " \
            f"t.bucketfs_conn_name, " \
            f"t.sub_dir, " \
            f"t.model_name, " \
            f"t.first_text, " \
            f"t.second_text" \
            f") FROM (VALUES {python_rows_to_sql(input_data)} " \
            f"AS t(device_id, bucketfs_conn_name, sub_dir, " \
            f"model_name, first_text, second_text));"

    # execute sequence classification UDF
    result = pyexasol_connection.execute(query).fetchall()

    # assertions
    assert result[0][-1] is None
    added_columns = 3  # label,score,error_message
    removed_columns = 1  # device_id
    n_rows_result = n_rows * n_labels
    n_cols_result = len(input_data[0]) + (added_columns - removed_columns)
    assert len(result) == n_rows_result and len(result[0]) == n_cols_result
