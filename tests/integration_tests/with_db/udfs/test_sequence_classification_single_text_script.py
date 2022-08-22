import pytest
from tests.utils.parameters import model_params


@pytest.mark.parametrize("upload_model_to_bucketfs", [model_params.base],
                         indirect=["upload_model_to_bucketfs"])
def test_sequence_classification_single_text_script(
        upload_language_container, setup_database,
        pyexasol_connection, upload_model_to_bucketfs):

    bucketfs_conn_name, schema_name = setup_database
    n_labels = 2
    n_rows = 100
    input_data = []
    for i in range(n_rows):
        input_data.append((
            '',
            bucketfs_conn_name,
            str(model_params.sub_dir),
            model_params.base,
            model_params.text_data))

    query = f"SELECT TE_SEQUENCE_CLASSIFICATION_SINGLE_TEXT_UDF(" \
            f"t.device_id, " \
            f"t.bucketfs_conn_name, " \
            f"t.sub_dir, " \
            f"t.model_name, " \
            f"t.text_data) " \
            f"FROM (VALUES {str(tuple(input_data))} " \
            f"AS t(device_id, bucketfs_conn_name, " \
            f"sub_dir, model_name, text_data));"

    # execute sequence classification UDF
    result = pyexasol_connection.execute(query).fetchall()
    print(result)

    # assertions
    n_rows_result = n_rows * n_labels
    n_cols_result = len(input_data[0]) + 1  # + 2 new cols -1 device_id col
    assert len(result) == n_rows_result and len(result[0]) == n_cols_result


