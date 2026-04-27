CREATE OR REPLACE {{ language_alias }} SET SCRIPT "AI_SENTIMENT"(
    text_data VARCHAR(2000000)
)EMITS (
    text_data VARCHAR(2000000),
    label VARCHAR(2000000),
    score DOUBLE,
    error_message VARCHAR(2000000) ) AS

{{ script_content }}

/