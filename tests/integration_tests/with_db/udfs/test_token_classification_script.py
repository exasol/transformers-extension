from tests.integration_tests.with_db.udfs.python_rows_to_sql import python_rows_to_sql
from tests.utils.parameters import model_params
import pytest


@pytest.mark.skip('Debugging')
def test_token_classification_script(
        setup_database, db_conn, upload_base_model_to_bucketfs):
    bucketfs_conn_name, _ = setup_database
    aggregation_strategy = "simple"
    n_rows = 100
    input_data = []
    for i in range(n_rows):
        input_data.append((
            '',
            bucketfs_conn_name,
            str(model_params.sub_dir),
            model_params.base_model_specs.model_name,
            model_params.text_data,
            aggregation_strategy
        ))

    query = f"SELECT TE_TOKEN_CLASSIFICATION_UDF(" \
            f"t.device_id, " \
            f"t.bucketfs_conn_name, " \
            f"t.sub_dir, " \
            f"t.model_name, " \
            f"t.text_data, " \
            f"t.aggregation_strategy" \
            f") FROM (VALUES {python_rows_to_sql(input_data)} " \
            f"AS t(device_id, bucketfs_conn_name, " \
            f"sub_dir, model_name, text_data, aggregation_strategy));"

    # execute sequence classification UDF
    result = db_conn.execute(query).fetchall()

    # assertions
    assert result[0][-1] is None
    added_columns = 6  # start_pos,end_pos,word,entity,score,error_message
    removed_columns = 1  # device_id
    n_cols_result = len(input_data[0]) + (added_columns - removed_columns)
    assert len(result) >= n_rows and len(result[0]) == n_cols_result
