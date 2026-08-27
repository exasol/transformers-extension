from noxconfig import PROJECT_CONFIG


def test_onprem_integration_test_files_are_sorted_and_root_relative():
    assert PROJECT_CONFIG.onprem_integration_test_files == [
        "test/integration_tests/with_db/deployment/test_deploy_cli.py",
        "test/integration_tests/with_db/deployment/test_install_default_models.py",
        "test/integration_tests/with_db/deployment/test_scripts_deployer.py",
        "test/integration_tests/with_db/deployment/test_scripts_deployer_cli.py",
        "test/integration_tests/with_db/test_upload_model.py",
        "test/integration_tests/with_db/udfs/test_ai_answer_extended_script.py",
        "test/integration_tests/with_db/udfs/test_ai_answer_script.py",
        "test/integration_tests/with_db/udfs/test_ai_classify_extended_script.py",
        "test/integration_tests/with_db/udfs/test_ai_classify_script.py",
        "test/integration_tests/with_db/udfs/test_ai_complete_extended_script.py",
        (
            "test/integration_tests/with_db/udfs/"
            "test_ai_custom_classify_extended_script.py"
        ),
        "test/integration_tests/with_db/udfs/test_ai_entailment_extended_script.py",
        "test/integration_tests/with_db/udfs/test_ai_extract_entities_script.py",
        "test/integration_tests/with_db/udfs/test_ai_extract_extended_script.py",
        "test/integration_tests/with_db/udfs/test_ai_fill_mask_extended_script.py",
        "test/integration_tests/with_db/udfs/test_ai_sentiment_script.py",
        "test/integration_tests/with_db/udfs/test_ai_translate_extended_script.py",
        "test/integration_tests/with_db/udfs/test_ai_translate_script.py",
        "test/integration_tests/with_db/udfs/test_delete_model.py",
        "test/integration_tests/with_db/udfs/test_ls_models_script.py",
        "test/integration_tests/with_db/udfs/test_model_downloader_udf_script.py",
        "test/integration_tests/with_db/udfs/test_prediction_with_downloader_udf.py",
        "test/integration_tests/with_db/utils/test_model_utils.py",
    ]
