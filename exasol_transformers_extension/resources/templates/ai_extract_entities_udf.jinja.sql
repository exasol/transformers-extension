CREATE OR REPLACE {{ language_alias }} SET SCRIPT "AI_EXTRACT_ENTITIES"(
    text_data VARCHAR(2000000),
    ORDER BY {{ ordered_columns | join(" ASC,") }} ASC
)EMITS (
    text_data VARCHAR(2000000),
    start_pos INTEGER,
    end_pos INTEGER,
    word VARCHAR(2000000),
    entity VARCHAR(2000000),
    score DOUBLE,
    error_message VARCHAR(2000000) ) AS

{{ script_content }}

/