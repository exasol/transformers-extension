import pytest

from exasol_transformers_extension.deployment.default_udf_parameters import DEFAULT_BUCKETFS_CONN_NAME

from test.integration_tests.with_db.udfs.test_ai_translate_extended_script import run_ai_translate_extended_script_test


@pytest.mark.parametrize(
    "bfs_conn_name",
    [
        "non-existing-con-name",
        DEFAULT_BUCKETFS_CONN_NAME
    ],
)
def test_bfs_conn_not_exist(
    setup_database, db_conn, upload_translation_model_to_bucketfs, bfs_conn_name
):
    n_rows = 3
    result, _ = run_ai_translate_extended_script_test(bfs_conn_name, n_rows, db_conn)

    print(result[0][-1])

    if bfs_conn_name != DEFAULT_BUCKETFS_CONN_NAME:
        expected_error_msg = ("The given bucketfs connection by the name of non-existing-con-name does not exist. "
                     "Either us another connection, or create it in the Exasol Database.")

        # assertions
        assert expected_error_msg in result[0][-1]

    else:
        expected_error_msg = ("In order to use this UDF, a BucketFSConnection by the name '{DEFAULT_BUCKETFS_CONN_NAME}' "
                "must be created in the Exasol Database.".format(DEFAULT_BUCKETFS_CONN_NAME=DEFAULT_BUCKETFS_CONN_NAME))

        # assertions
        assert expected_error_msg in result[0][-1]

