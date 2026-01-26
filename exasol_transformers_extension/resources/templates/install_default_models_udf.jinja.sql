CREATE OR REPLACE {{ language_alias }} SET SCRIPT "TE_INSTALL_DEFAULT_MODEL_UDF"(
) EMITS (
    sub_dir VARCHAR(2000000),
    model_name VARCHAR(2000000),
    task_type VARCHAR(2000000),
    success BOOLEAN,
    error_message VARCHAR(2000000)
) AS

{{ script_content }}

/