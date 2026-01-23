from test.integration_tests.utils.model_output_quality_checkers import (
    assert_lenient_check_of_output_quality_with_score,
)
from test.integration_tests.utils.model_output_result_number_checker import (
    assert_correct_number_of_results,
)
from test.integration_tests.with_db.udfs.python_rows_to_sql import python_rows_to_sql
from test.utils.parameters import model_params

import pytest


@pytest.mark.parametrize(
    "return_ranks, number_results_per_input",
    [("ALL", None), ("HIGHEST", 1)],
)
def test_ai_entailment_extended_script(
    return_ranks,
    number_results_per_input,
    setup_database,
    db_conn,
    upload_text_classification_pair_model_to_bucketfs,
):
    bucketfs_conn_name, _ = setup_database
    n_labels = 3

    if not number_results_per_input:
        number_results_per_input = n_labels

    n_rows = 100

    input_data = []
    for i in range(n_rows):
        input_data.append(
            (
                "",
                bucketfs_conn_name,
                str(model_params.sub_dir),
                model_params.text_classification_pair_model_specs.model_name,
                "The database software company Exasol is based in Nuremberg",
                "The main Exasol office is located in Flensburg",
                return_ranks,
            )
        )

    query = (
        f"SELECT AI_ENTAILMENT_EXTENDED("
        f"t.device_id, "
        f"t.bucketfs_conn_name, "
        f"t.sub_dir, "
        f"t.model_name, "
        f"t.first_text, "
        f"t.second_text, "
        f"t.return_ranks"
        f") FROM (VALUES {python_rows_to_sql(input_data)} "
        f"AS t(device_id, bucketfs_conn_name, sub_dir, "
        f"model_name, first_text, second_text, return_ranks));"
    )

    # execute ai_entailment_extended UDF
    result = db_conn.execute(query).fetchall()

    # assertions
    assert result[0][-1] is None

    n_rows_result = n_rows * number_results_per_input
    # added_columns: label,score,rank,error_message
    # removed_columns: device_id,
    assert_correct_number_of_results(4, 1, input_data[0], result, n_rows_result)

    # Since in this test the input is two sentences which contradict each other, which the test model can detect,
    # the "acceptable_results" here is the label "contradiction" with a reasonably high score.
    # possible labels: contradiction, entailment, neutral
    acceptable_results = ["contradiction"]
    assert_lenient_check_of_output_quality_with_score(
        result, acceptable_results, 1 / 1.5, label_index=6
    )
