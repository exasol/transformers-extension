from tests.utils.parameters import model_params


def test_model_downloader_udf_script(
        upload_language_container,
        setup_database,
        pyexasol_connection, upload_model_to_bucketfs):

    bucketfs_conn_name, schema_name = setup_database

    # execute sequence classification UDF
    result = pyexasol_connection.execute(
        f"SELECT TE_SEQUENCE_CLASSIFICATION_SINGLE_TEXT_UDF("
        f"'{bucketfs_conn_name}', "
        f"'{model_params.name}', "
        f"'{model_params.text_data}')"
    ).fetchall()

    # assertions
    assert result

