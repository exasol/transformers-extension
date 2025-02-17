from test.integration_tests.with_db.udfs.python_rows_to_sql import python_rows_to_sql
from test.utils.parameters import model_params


def test_translation_script(
        setup_database, db_conn, upload_seq2seq_model_to_bucketfs):
    bucketfs_conn_name, _ = setup_database
    n_rows = 100
    src_lang = "English"
    target_lang = "German"
    max_length = 50
    input_data = []
    for i in range(n_rows):
        input_data.append((
            '',
            bucketfs_conn_name,
            str(model_params.sub_dir),
            model_params.seq2seq_model_specs.model_name,
            'The database software company Exasol is based in Nuremberg',
            src_lang,
            target_lang,
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
            f") FROM (VALUES {python_rows_to_sql(input_data)} " \
            f"AS t(device_id, bucketfs_conn_name, sub_dir, model_name, " \
            f"text_data, source_language, target_language, max_length));"

    # execute sequence classification UDF
    result = db_conn.execute(query).fetchall()

    # assertions
    assert result[0][-1] is None
    added_columns = 2  # translation_text,error_message
    removed_columns = 1  # device_id
    n_rows_result = n_rows
    n_cols_result = len(input_data[0]) + (added_columns - removed_columns)
    assert len(result) == n_rows_result and len(result[0]) == n_cols_result

    # lenient test for quality of results, will be replaced by deterministic test later
    results = [result[i][7] for i in range(len(result))]
    acceptable_results = ["Die Datenbanksoftware Exasol hat ihren Sitz in Nürnberg"]
    number_accepted_results = 0

    def contains(string,list):
        return any(map(lambda x: x in string, list))

    for i in range(len(results)):
        if contains(results[i], acceptable_results):
            number_accepted_results += 1
    assert number_accepted_results > n_rows_result/2
