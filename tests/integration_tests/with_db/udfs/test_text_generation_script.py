from tests.fixtures.model_fixture import upload_text_generation_model_to_bucketfs
from tests.integration_tests.with_db.udfs.python_rows_to_sql import python_rows_to_sql
from tests.utils.parameters import model_params

#for debug
from tests.fixtures.model_fixture import *
from tests.fixtures.setup_database_fixture import *
from tests.fixtures.language_container_fixture import *
from tests.fixtures.bucketfs_fixture import *
from tests.fixtures.database_connection_fixture import *

def test_text_generation_script(
        setup_database, db_conn, upload_text_generation_model_to_bucketfs):
    bucketfs_conn_name, _ = setup_database
    text_data = "Exasol is an analytics database management"
    n_rows = 100
    max_length = 12
    return_full_text = True
    input_data = []
    for i in range(n_rows):
        input_data.append((
            '',
            bucketfs_conn_name,
            str(model_params.sub_dir),
            model_params.text_gen_model_specs.model_name,
            text_data,
            max_length,
            return_full_text
        ))

    query = f"SELECT TE_TEXT_GENERATION_UDF(" \
            f"t.device_id, " \
            f"t.bucketfs_conn_name, " \
            f"t.sub_dir, " \
            f"t.model_name, " \
            f"t.text_data, " \
            f"t.max_length," \
            f"t.return_full_text" \
            f") FROM (VALUES {python_rows_to_sql(input_data)} " \
            f"AS t(device_id, bucketfs_conn_name, sub_dir, " \
            f"model_name, text_data, max_length, return_full_text));"

    # execute sequence classification UDF
    result = db_conn.execute(query).fetchall()

    # assertions
    assert result[0][-1] is None
    added_columns = 2  # generated_text,error_message
    removed_columns = 1  # device_id
    n_rows_result = n_rows
    n_cols_result = len(input_data[0]) + (added_columns - removed_columns)
    assert len(result) == n_rows_result and len(result[0]) == n_cols_result

    # lenient test for quality of results, will be replaced by deterministic test later
    for i in range(5):
        print(result[i])
    results = [result[i][6] for i in range(len(result))]
    acceptable_results = ["software", "system", "solution", "tool"]
    number_accepted_results = 0
    def contains(string,list):
        return any(map(lambda x: x in string, list))

    for i in range(len(results)):
        if contains(results[i], acceptable_results):
            number_accepted_results += 1
    assert number_accepted_results > n_rows_result/2
