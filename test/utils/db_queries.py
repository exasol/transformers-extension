import dataclasses
from typing import List

expected_script_list_without_span = [
    "TE_MODEL_DOWNLOADER_UDF",
    "TE_DELETE_MODEL_UDF",
    "AI_CUSTOM_CLASSIFY_EXTENDED",
    "AI_ENTAILMENT_EXTENDED",
    "AI_ANSWER_EXTENDED",
    "AI_FILL_MASK_EXTENDED",
    "AI_COMPLETE_EXTENDED",
    "AI_EXTRACT_EXTENDED",
    "AI_TRANSLATE_EXTENDED",
    "AI_CLASSIFY_EXTENDED",
    "TE_LIST_MODELS_UDF",
    "TE_INSTALL_DEFAULT_MODEL_UDF",
]

expected_script_list_with_span = [
    "TE_MODEL_DOWNLOADER_UDF",
    "TE_DELETE_MODEL_UDF",
    "AI_CUSTOM_CLASSIFY_EXTENDED",
    "AI_ENTAILMENT_EXTENDED",
    "AI_ANSWER_EXTENDED",
    "AI_FILL_MASK_EXTENDED",
    "AI_COMPLETE_EXTENDED",
    "AI_EXTRACT_EXTENDED_WITH_SPAN",
    "AI_TRANSLATE_EXTENDED",
    "AI_CLASSIFY_EXTENDED_WITH_SPAN",
    "TE_LIST_MODELS_UDF",
    "TE_INSTALL_DEFAULT_MODEL_UDF",
]

expected_script_list_all = [
    "TE_MODEL_DOWNLOADER_UDF",
    "TE_DELETE_MODEL_UDF",
    "AI_EXTRACT_EXTENDED",
    "AI_CUSTOM_CLASSIFY_EXTENDED",
    "AI_ENTAILMENT_EXTENDED",
    "AI_ANSWER_EXTENDED",
    "AI_FILL_MASK_EXTENDED",
    "AI_COMPLETE_EXTENDED",
    "AI_EXTRACT_EXTENDED_WITH_SPAN",
    "AI_TRANSLATE_EXTENDED",
    "AI_CLASSIFY_EXTENDED",
    "AI_CLASSIFY_EXTENDED_WITH_SPAN",
    "TE_LIST_MODELS_UDF",
    "TE_INSTALL_DEFAULT_MODEL_UDF",
]


@dataclasses.dataclass
class ExaParameter:
    system_value: str
    session_value: str


class DBQueries:
    @staticmethod
    def get_all_scripts(db_conn, schema_name) -> list[int]:
        query_all_scripts = f"""
                SELECT SCRIPT_NAME
                FROM EXA_ALL_SCRIPTS
                WHERE SCRIPT_SCHEMA = '{schema_name}'
            """
        all_scripts = db_conn.execute(query_all_scripts).fetchall()
        return list(map(lambda x: x[0], all_scripts))

    @staticmethod
    def check_all_scripts_deployed(db_conn, schema_name, expected_script_list) -> bool:
        all_scripts = DBQueries.get_all_scripts(db_conn, schema_name)
        return set(all_scripts) == set(expected_script_list)

    @staticmethod
    def get_language_settings(db_conn) -> ExaParameter:
        query = """
            SELECT "SYSTEM_VALUE", "SESSION_VALUE"
            FROM SYS.EXA_PARAMETERS
            WHERE PARAMETER_NAME='SCRIPT_LANGUAGES'"""
        result = db_conn.execute(query).fetchall()
        if len(result) != 1:
            raise RuntimeError(
                f"Got not exactly one row for the SCRIPT_LANGUAGES parameter. Got {result}"
            )
        exa_parameter = ExaParameter(
            system_value=result[0][0], session_value=result[0][1]
        )
        return exa_parameter
