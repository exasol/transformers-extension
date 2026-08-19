CREATE OR REPLACE {{ language_alias }} SET SCRIPT "AI_TRANSLATE"(
    text_data VARCHAR(2000000),
    source_language VARCHAR(2000000),
    target_language VARCHAR(2000000)
)EMITS (
    text_data VARCHAR(2000000),
    source_language VARCHAR(2000000),
    target_language VARCHAR(2000000),
    translation_text VARCHAR(2000000),
    error_message VARCHAR(2000000)) AS

{{ script_content }}

/