from noxconfig import PROJECT_CONFIG


def test_saas_integration_test_files_allowlist() -> None:
    assert PROJECT_CONFIG.saas_integration_test_files == [
        "test/integration_tests/with_db/deployment/test_deploy_cli.py",
        "test/integration_tests/with_db/test_upload_model.py",
        "test/integration_tests/with_db/deployment/test_install_default_models.py",
        "test/integration_tests/with_db/udfs/test_model_downloader_udf_script.py",
        "test/integration_tests/with_db/udfs/test_prediction_with_downloader_udf.py",
        "test/integration_tests/with_db/udfs/test_delete_model.py",
        "test/integration_tests/with_db/udfs/test_ls_models_script.py",
        "test/integration_tests/with_db/udfs/test_ai_sentiment_script.py",
        "test/integration_tests/with_db/udfs/test_ai_answer_extended_script.py",
    ]
