CREATE OR REPLACE {{ language_alias }} SET SCRIPT "TE_SEQUENCE_CLASSIFICATION_SINGLE_TEXT_UDF"(
    bucketfs_conn VARCHAR(2000000),
    model_name VARCHAR(2000000),
    text VARCHAR(2000000)
)EMITS  (outputs VARCHAR(2000000)) AS

{{ script_content }}

/