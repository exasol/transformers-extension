from test.integration_tests.utils.model_output_quality_checkers import (
    assert_lenient_check_of_output_quality,
)
from test.integration_tests.utils.model_output_result_number_checker import (
    assert_correct_number_of_results,
)
from test.integration_tests.with_db.udfs.python_rows_to_sql import python_rows_to_sql
from test.utils.parameters import model_params


def test_ai_translate_extended_script(
    setup_database, db_conn, upload_translation_model_to_bucketfs
):
    bucketfs_conn_name, _ = setup_database
    n_rows = 100
    src_lang = "English"
    target_lang = "German"
    max_new_tokens = 50
    input_data = []
    for i in range(n_rows):
        input_data.append(
            (
                "",
                bucketfs_conn_name,
                str(model_params.sub_dir),
                model_params.seq2seq_model_specs.model_name,
                "The database software company Exasol is based in Nuremberg",
                src_lang,
                target_lang,
                max_new_tokens,
            )
        )

    query = (
        f"SELECT AI_TRANSLATE_EXTENDED("
        f"t.device_id, "
        f"t.bucketfs_conn_name, "
        f"t.sub_dir, "
        f"t.model_name, "
        f"t.text_data, "
        f"t.source_language, "
        f"t.target_language, "
        f"t.max_new_tokens"
        f") FROM (VALUES {python_rows_to_sql(input_data)} "
        f"AS t(device_id, bucketfs_conn_name, sub_dir, model_name, "
        f"text_data, source_language, target_language, max_new_tokens));"
    )

    # execute sequence classification UDF
    result = db_conn.execute(query).fetchall()

    # assertions
    assert result[0][-1] is None
    added_columns = 2  # translation_text,error_message
    removed_columns = 1  # device_id
    assert_correct_number_of_results(
        added_columns, removed_columns, input_data[0], result, n_rows
    )

    acceptable_results = ["Die Datenbanksoftware Exasol hat ihren Sitz in Nürnberg"]
    assert_lenient_check_of_output_quality(
        result, acceptable_results, acceptance_factor=0.5, label_index=7
    )
