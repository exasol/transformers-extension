from typing import List

deployed_script_list = [
    "TE_MODEL_DOWNLOADER_UDF",
]


class DBQueries:
    @staticmethod
    def get_all_scripts(db_conn, schema_name) -> List[int]:
        query_all_scripts = \
            f"""
                SELECT SCRIPT_NAME 
                FROM EXA_ALL_SCRIPTS
                WHERE SCRIPT_SCHEMA = '{schema_name.upper()}'
            """
        all_scripts = db_conn.execute(query_all_scripts).fetchall()
        return list(map(lambda x: x[0], all_scripts))

    @staticmethod
    def check_all_scripts_deployed(db_conn, schema_name) -> bool:
        all_scripts = DBQueries.get_all_scripts(
            db_conn, schema_name)
        return all(script in all_scripts for script in deployed_script_list)

    @staticmethod
    def get_language_settings(db_conn) -> List:
        query = f"""
            SELECT "SYSTEM_VALUE", "SESSION_VALUE" 
            FROM SYS.EXA_PARAMETERS 
            WHERE PARAMETER_NAME='SCRIPT_LANGUAGES'"""
        return db_conn.execute(query).fetchall()
