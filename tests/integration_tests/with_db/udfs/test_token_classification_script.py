from tests.integration_tests.with_db.udfs.python_rows_to_sql import python_rows_to_sql
from tests.utils.parameters import model_params


def setup_common_input_data():
    aggregation_strategy = "simple"
    n_rows = 100
    text_data = 'The database software company Exasol is based in Nuremberg'
    return aggregation_strategy, n_rows, text_data

def correct_number_of_results(added_columns: int, removed_columns: int,
                              input_data_row: tuple, result: list, n_rows: int):
    n_cols_result = len(input_data_row) + (added_columns - removed_columns)
    return len(result) >= n_rows and len(result[0]) == n_cols_result

def lenient_check_of_output_quality(results: list):
    acceptable_result_sets = [["Exasol", "ORG"], ["Nuremberg", "LOC"]]
    number_accepted_results = 0

    for i in range(len(results)):
        if results[i] in acceptable_result_sets:
            number_accepted_results += 1
    assert number_accepted_results > len(results)/1.5

def test_token_classification_script_without_spans(
        setup_database, db_conn, upload_token_classification_model_to_bucketfs):
    bucketfs_conn_name, schema_name = setup_database
    aggregation_strategy, n_rows, text_data = setup_common_input_data()
    input_data = []
    for i in range(n_rows):
        input_data.append((
            '',
            bucketfs_conn_name,
            str(model_params.sub_dir),
            model_params.token_model_specs.model_name,
            text_data,
            aggregation_strategy
        ))

    query = f"SELECT {schema_name}.TE_TOKEN_CLASSIFICATION_UDF(" \
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
    # added_columns: start_pos,end_pos,word,entity,score,error_message
    # removed_columns: device_id
    assert correct_number_of_results(6, 1,
                                     input_data[0], result, n_rows)

    # lenient test for quality of results, will be replaced by deterministic test later
    results = [[result[i][7], result[i][8]] for i in range(len(result))]
    assert lenient_check_of_output_quality(results)

def test_token_classification_script_with_span(
        setup_database, db_conn, upload_token_classification_model_to_bucketfs):
    bucketfs_conn_name, schema_name = setup_database
    aggregation_strategy, n_rows, text_data = setup_common_input_data()
    input_data = []
    for i in range(n_rows):
        input_data.append((
            '',
            bucketfs_conn_name,
            str(model_params.sub_dir),
            model_params.token_model_specs.model_name,
            text_data,
            i,
            0,
            len(text_data),
            aggregation_strategy
        ))

    query = f"SELECT {schema_name}.TE_TOKEN_CLASSIFICATION_UDF_WITH_SPAN(" \
            f"t.device_id, " \
            f"t.bucketfs_conn_name, " \
            f"t.sub_dir, " \
            f"t.model_name, " \
            f"t.text_data, " \
            f"t.text_data_docid, " \
            f"t.text_data_char_begin, "\
            f"t.text_data_char_end, " \
            f"t.aggregation_strategy" \
            f") FROM (VALUES {python_rows_to_sql(input_data)} " \
            f"AS t(device_id, bucketfs_conn_name, " \
            f"sub_dir, model_name, text_data, text_data_docid, text_data_char_begin, " \
            f"text_data_char_end, aggregation_strategy));"

    # execute sequence classification UDF
    result = db_conn.execute(query).fetchall()
    # assertions
    assert result[0][-1] is None
    # added_columns: entity_covered_text, entity_type, score, entity_docid, entity_char_begin, entity_char_end, error_message
    # removed_columns: # device_id, text_data, text_data_docid, text_data_char_begin, text_data_char_end
    assert correct_number_of_results(7, 5,
                                     input_data[0], result, n_rows)
    # lenient test for quality of results, will be replaced by deterministic test later
    results = [[result[i][4], result[i][5]] for i in range(len(result))]
    assert lenient_check_of_output_quality(results)