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

    # lenient test for quality of results, will be replaced by deterministic test later
    number_accepted_results = 0
    for i in range(len(result)):
        if (
            result[i][5]
            == "contradiction"  # possible labels: contradiction, entailment, neutral
            and result[i][6] > 0.8
        ):  # check if confidence resonably high
            number_accepted_results += 1
        elif result[i][6] < 0.2:
            number_accepted_results += 1
    assert number_accepted_results > n_rows_result / 1.5
