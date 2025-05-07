import dataclasses
from typing import List

expected_script_list_without_span = [
    "TE_MODEL_DOWNLOADER_UDF",
    "TE_SEQUENCE_CLASSIFICATION_SINGLE_TEXT_UDF",
    "TE_SEQUENCE_CLASSIFICATION_TEXT_PAIR_UDF",
    "TE_QUESTION_ANSWERING_UDF",
    "TE_FILLING_MASK_UDF",
    "TE_TEXT_GENERATION_UDF",
    "TE_TOKEN_CLASSIFICATION_UDF",
    "TE_TRANSLATION_UDF",
    "TE_ZERO_SHOT_TEXT_CLASSIFICATION_UDF",
]

expected_script_list_with_span = [
    "TE_MODEL_DOWNLOADER_UDF",
    "TE_SEQUENCE_CLASSIFICATION_SINGLE_TEXT_UDF",
    "TE_SEQUENCE_CLASSIFICATION_TEXT_PAIR_UDF",
    "TE_QUESTION_ANSWERING_UDF",
    "TE_FILLING_MASK_UDF",
    "TE_TEXT_GENERATION_UDF",
    "TE_TOKEN_CLASSIFICATION_UDF_WITH_SPAN",
    "TE_TRANSLATION_UDF",
    "TE_ZERO_SHOT_TEXT_CLASSIFICATION_UDF_WITH_SPAN",
]

expected_script_list_all = [
    "TE_MODEL_DOWNLOADER_UDF",
    "TE_TOKEN_CLASSIFICATION_UDF",
    "TE_SEQUENCE_CLASSIFICATION_SINGLE_TEXT_UDF",
    "TE_SEQUENCE_CLASSIFICATION_TEXT_PAIR_UDF",
    "TE_QUESTION_ANSWERING_UDF",
    "TE_FILLING_MASK_UDF",
    "TE_TEXT_GENERATION_UDF",
    "TE_TOKEN_CLASSIFICATION_UDF_WITH_SPAN",
    "TE_TRANSLATION_UDF",
    "TE_ZERO_SHOT_TEXT_CLASSIFICATION_UDF",
    "TE_ZERO_SHOT_TEXT_CLASSIFICATION_UDF_WITH_SPAN",
]


@dataclasses.dataclass
class ExaParameter:
    system_value: str
    session_value: str


class DBQueries:
    @staticmethod
    def get_all_scripts(db_conn, schema_name) -> List[int]:
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
