CREATE OR REPLACE {{ language_alias }} SET SCRIPT "AI_ANSWER"(
    question VARCHAR(2000000),
    context_text VARCHAR(2000000)
    ORDER BY {{ ordered_columns | join(" ASC,") }} ASC
)EMITS (
    question VARCHAR(2000000),
    context_text VARCHAR(2000000),
    answer VARCHAR(2000000),
    error_message VARCHAR(2000000) ) AS

{{ script_content }}

/