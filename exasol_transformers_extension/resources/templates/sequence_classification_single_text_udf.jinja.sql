CREATE OR REPLACE {{ language_alias }} SET SCRIPT "TE_SEQUENCE_CLASSIFICATION_SINGLE_TEXT_UDF"(
    bucketfs_conn VARCHAR(2000000),
    model_name VARCHAR(2000000),
    text_data VARCHAR(2000000)
)EMITS  (...) AS

{{ script_content }}

/