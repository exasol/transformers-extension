import pytest
from tests.utils.parameters import model_params


@pytest.mark.parametrize("upload_model_to_bucketfs", [model_params.seq2seq],
                         indirect=["upload_model_to_bucketfs"])
def test_translation_script(
        upload_language_container, setup_database,
        pyexasol_connection, upload_model_to_bucketfs):

    bucketfs_conn_name, schema_name = setup_database
    n_rows = 100
    model_name = "t5-small"
    src_lang = "English"
    tgt_lang = "German"
    max_length = 50
    input_data = []
    for i in range(n_rows):
        input_data.append((
            '',
            bucketfs_conn_name,
            str(model_params.sub_dir),
            model_name,
            model_params.text_data,
            src_lang,
            tgt_lang,
            max_length
        ))

    query = f"SELECT TE_TRANSLATION_UDF(" \
            f"t.device_id, " \
            f"t.bucketfs_conn_name, " \
            f"t.sub_dir, " \
            f"t.model_name, " \
            f"t.text_data, " \
            f"t.source_language, " \
            f"t.target_language, " \
            f"t.max_length" \
            f") FROM (VALUES {str(tuple(input_data))} " \
            f"AS t(device_id, bucketfs_conn_name, sub_dir, model_name, " \
            f"text_data, source_language, target_language, max_length));"

    # execute sequence classification UDF
    result = pyexasol_connection.execute(query).fetchall()
    print(result)

    # assertions
    n_rows_result = n_rows
    n_cols_result = len(input_data[0]) + 0  # + 1 new cols -1 device_id col
    assert len(result) == n_rows_result and len(result[0]) == n_cols_result

