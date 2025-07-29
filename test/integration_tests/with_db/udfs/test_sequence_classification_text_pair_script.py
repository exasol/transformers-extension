from test.integration_tests.with_db.udfs.python_rows_to_sql import python_rows_to_sql
from test.utils.parameters import model_params


def test_sequence_classification_text_pair_script(
    setup_database, db_conn, upload_sequence_classification_pair_model_to_bucketfs
):
    bucketfs_conn_name, _ = setup_database
    n_labels = 3
    n_rows = 100
    input_data = []
    for i in range(n_rows):
        input_data.append(
            (
                "",
                bucketfs_conn_name,
                str(model_params.sub_dir),
                model_params.sequence_class_pair_model_specs.model_name,
                "The database software company Exasol is based in Nuremberg",
                "The main Exasol office is located in Flensburg",
            )
        )

    query = (
        f"SELECT TE_SEQUENCE_CLASSIFICATION_TEXT_PAIR_UDF("
        f"t.device_id, "
        f"t.bucketfs_conn_name, "
        f"t.sub_dir, "
        f"t.model_name, "
        f"t.first_text, "
        f"t.second_text"
        f") FROM (VALUES {python_rows_to_sql(input_data)} "
        f"AS t(device_id, bucketfs_conn_name, sub_dir, "
        f"model_name, first_text, second_text));"
    )

    # execute sequence classification UDF
    result = db_conn.execute(query).fetchall()

    # assertions
    assert result[0][-1] is None
    added_columns = 3  # label,score,error_message
    removed_columns = 1  # device_id
    n_rows_result = n_rows * n_labels
    n_cols_result = len(input_data[0]) + (added_columns - removed_columns)
    assert len(result) == n_rows_result and len(result[0]) == n_cols_result

    # lenient test for quality of results,
    # checks whether enough of the results are of "good quality".
    # Since in this test the input is two sentences which contradict each other, which the test model can detect,
    # the "acceptable_results" here is the label "contradiction" with a reasonably high score.
    # we want high confidence on good results, and low confidence on bad results. however, cutoffs for
    # high and low confidence where not set in an elaborate scientific way.
    # this check is only here to assure us the models output is not totally of kilter
    # (and crucially does not get worse with our changes over time),
    # and therefore we can assume model loading and execution is working correctly.
    # a plan to make this check deterministic in the future exists.

    # An accepted results is defined as follows:
    #                       | label acceptable  | label unacceptable
    # --------------------------------------------------------------
    # high confidence       | acceptable        |  bad result
    # (result_score > 0.8)  |                   |
    # --------------------------------------------------------------
    # other confidence      | result not        |  result not
    # (result_score between | good enough to    |  good enough to
    # high and low)         | be accepted       |  be accepted
    # --------------------------------------------------------------
    # low confidence        | bad result        |  acceptable
    # (result_score < 0.2)  |                   |

    # we only sum up acceptable results below, because we already know we
    # have the correct number of results from the other checks.
    number_accepted_results = 0
    for i in range(len(result)):
        result_score = result[i][6]
        result_label = result[i][5]
        if (
            result_label
            == "contradiction"  # possible labels: contradiction, entailment, neutral
            and result_score > 0.8
        ):  # check if confidence reasonably high
            number_accepted_results += 1
        elif result_score < 0.2 and result_label != "contradiction":
            number_accepted_results += 1
    assert number_accepted_results > n_rows_result / 1.5
