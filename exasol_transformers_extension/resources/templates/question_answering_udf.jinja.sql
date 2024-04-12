CREATE OR REPLACE {{ language_alias }} SET SCRIPT "TE_QUESTION_ANSWERING_UDF"(
    device_id INTEGER,
    bucketfs_conn VARCHAR(2000000),
    sub_dir VARCHAR(2000000),
    model_name VARCHAR(2000000),
    question VARCHAR(2000000),
    context_text VARCHAR(2000000),
    top_k INTEGER
    ORDER BY {{ ordered_columns | join(" ASC,") }} ASC
)EMITS (
    bucketfs_conn VARCHAR(2000000),
    sub_dir VARCHAR(2000000),
    model_name VARCHAR(2000000),
    question VARCHAR(2000000),
    context_text VARCHAR(2000000),
    top_k INTEGER,
    answer VARCHAR(2000000),
    score DOUBLE,
    rank INTEGER,
    error_message VARCHAR(2000000) ) AS

{{ script_content }}

/