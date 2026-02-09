CREATE OR REPLACE {{ language_alias }} SET SCRIPT "TE_INSTALL_DEFAULT_MODEL_UDF"(
) EMITS (
    model_path_in_udfs VARCHAR(2000000),
    model_path_of_tar_file_in_bucketfs VARCHAR(2000000)
) AS

{{ script_content }}

/