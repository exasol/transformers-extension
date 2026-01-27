from test.integration_tests.utils.model_output_quality_checkers import (
    assert_lenient_check_of_output_quality_for_result_set,
)
from test.integration_tests.utils.model_output_result_number_checker import (
    assert_correct_number_of_results_multiple_results_per_input,
)
from test.integration_tests.with_db.udfs.python_rows_to_sql import python_rows_to_sql
from test.utils.parameters import model_params


def setup_common_input_data():
    aggregation_strategy = "simple"
    n_rows = 100
    text_data = "The database software company Exasol is based in Nuremberg"
    return aggregation_strategy, n_rows, text_data


def test_ai_extract_extended_script_without_spans(
    setup_database, db_conn, upload_token_classification_model_to_bucketfs
):
    bucketfs_conn_name, _ = setup_database
    aggregation_strategy, n_rows, text_data = setup_common_input_data()
    input_data = []
    for i in range(n_rows):
        input_data.append(
            (
                "",
                bucketfs_conn_name,
                str(model_params.sub_dir),
                model_params.token_model_specs.model_name,
                text_data,
                aggregation_strategy,
            )
        )

    query = (
        f"SELECT AI_EXTRACT_EXTENDED("
        f"t.device_id, "
        f"t.bucketfs_conn_name, "
        f"t.sub_dir, "
        f"t.model_name, "
        f"t.text_data, "
        f"t.aggregation_strategy"
        f") FROM (VALUES {python_rows_to_sql(input_data)} "
        f"AS t(device_id, bucketfs_conn_name, "
        f"sub_dir, model_name, text_data, aggregation_strategy));"
    )

    # execute sequence classification UDF
    result = db_conn.execute(query).fetchall()
    # assertions
    assert result[0][-1] is None
    # added_columns: start_pos,end_pos,word,entity,score,error_message
    # removed_columns: device_id
    assert_correct_number_of_results_multiple_results_per_input(
        6, 1, input_data[0], result, n_rows
    )

    # lenient test for quality of results, will be replaced by deterministic test later
    results = [[result[i][7], result[i][8]] for i in range(len(result))]

    # lenient test for quality of results, will be replaced by deterministic test later
    acceptable_result_sets = [["Exasol", "ORG"], ["Nuremberg", "LOC"]]
    assert_lenient_check_of_output_quality_for_result_set(
        result, acceptable_result_sets, acceptance_factor=0.5, label_index=7
    )


def test_ai_extract_extended_script_with_span(
    setup_database, db_conn, upload_token_classification_model_to_bucketfs
):
    bucketfs_conn_name, _ = setup_database
    aggregation_strategy, n_rows, text_data = setup_common_input_data()
    input_data = []
    for i in range(n_rows):
        input_data.append(
            (
                "",
                bucketfs_conn_name,
                str(model_params.sub_dir),
                model_params.token_model_specs.model_name,
                text_data,
                i,
                0,
                len(text_data),
                aggregation_strategy,
            )
        )

    query = (
        f"SELECT AI_EXTRACT_EXTENDED_WITH_SPAN("
        f"t.device_id, "
        f"t.bucketfs_conn_name, "
        f"t.sub_dir, "
        f"t.model_name, "
        f"t.text_data, "
        f"t.text_data_doc_id, "
        f"t.text_data_char_begin, "
        f"t.text_data_char_end, "
        f"t.aggregation_strategy"
        f") FROM (VALUES {python_rows_to_sql(input_data)} "
        f"AS t(device_id, bucketfs_conn_name, "
        f"sub_dir, model_name, text_data, text_data_doc_id, text_data_char_begin, "
        f"text_data_char_end, aggregation_strategy));"
    )

    # execute sequence classification UDF
    result = db_conn.execute(query).fetchall()
    # assertions
    assert result[0][-1] is None
    # added_columns: entity_covered_text, entity_type, score, entity_doc_id, entity_char_begin, entity_char_end, error_message
    # removed_columns: # device_id, text_data
    assert_correct_number_of_results_multiple_results_per_input(
        7, 2, input_data[0], result, n_rows
    )
    # lenient test for quality of results, will be replaced by deterministic test later
    acceptable_result_sets = [["Exasol", "ORG"], ["Nuremberg", "LOC"]]
    assert_lenient_check_of_output_quality_for_result_set(
        result, acceptable_result_sets, acceptance_factor=0.5, label_index=7
    )
