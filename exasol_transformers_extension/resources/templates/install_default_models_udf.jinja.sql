CREATE OR REPLACE {{ language_alias }} SCALAR SCRIPT "INSTALL_AI_DEFAULT_MODEL_UDF"(...)
       EMITS (
    model_path_in_udfs VARCHAR(2000000),
    model_path_of_tar_file_in_bucketfs VARCHAR(2000000)
) AS

{{ script_content }}

/
