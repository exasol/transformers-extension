from tests.fixtures.model_fixture import upload_token_classification_model_to_bucketfs
from tests.integration_tests.with_db.udfs.python_rows_to_sql import python_rows_to_sql
from tests.utils.parameters import model_params

from tests.fixtures.model_fixture import *
from tests.fixtures.setup_database_fixture import *
from tests.fixtures.language_container_fixture import *
from tests.fixtures.bucketfs_fixture import *
from tests.fixtures.database_connection_fixture import *

def test_token_classification_script(
        setup_database, db_conn, upload_token_classification_model_to_bucketfs):
    bucketfs_conn_name, _ = setup_database
    aggregation_strategy = "simple"
    n_rows = 100
    input_data = []
    text_data = 'The database software company Exasol is based in Nuremberg'
    for i in range(n_rows):
        input_data.append((
            '',
            bucketfs_conn_name,
            str(model_params.sub_dir),
            model_params.token_model_specs.model_name,
            text_data,
            str((0, len(text_data))),
            aggregation_strategy
        ))

    query = f"SELECT TE_TOKEN_CLASSIFICATION_UDF_WITH_SPAN(" \
            f"t.device_id, " \
            f"t.bucketfs_conn_name, " \
            f"t.sub_dir, " \
            f"t.model_name, " \
            f"t.text_data, " \
            f"t.span, " \
            f"t.aggregation_strategy" \
            f") FROM (VALUES {python_rows_to_sql(input_data)} " \
            f"AS t(device_id, bucketfs_conn_name, " \
            f"sub_dir, model_name, text_data, span, aggregation_strategy));"

    # execute sequence classification UDF
    result = db_conn.execute(query).fetchall()
    print(result) #should:['bucketfs_conn', 'sub_dir', 'model_name', 'text_data', 'span',
      # 'aggregation_strategy', 'start_pos', 'end_pos', 'word', 'entity',
      # 'score', 'token_span', 'error_message'],
    # assertions
    assert result[0][-1] is None
    added_columns = 7  # start_pos,end_pos,word,entity,score,token_span,error_message
    removed_columns = 1  # device_id
    n_cols_result = len(input_data[0]) + (added_columns - removed_columns)
    assert len(result) >= n_rows and len(result[0]) == n_cols_result

    # lenient test for quality of results, will be replaced by deterministic test later
    results = [[result[i][7], result[i][8]] for i in range(len(result))]
    acceptable_result_sets = [["Exasol", "ORG"], ["Nuremberg", "LOC"]]
    number_accepted_results = 0

    for i in range(len(results)):
        if results[i] in acceptable_result_sets:
            number_accepted_results += 1
    assert number_accepted_results > len(result)/1.5