"""
Writes SQL queries for the creation of all udf script to create_script.sql
"""

from pathlib import Path

from exasol_transformers_extension.deployment.script_deployment_queries import (
    ScriptDeploymentQueries,
)


def write_create_script():
    """
    Writes SQL queries for the creation of all udf script to create_script.sql
    """
    print("Write create script")
    sdq_creator = ScriptDeploymentQueries(
        language_alias="PYTHON3_TE", use_spans=True, install_all_scripts=True
    )  # todo wich udfs do we want to deploy with this script?

    root_dir = Path(__file__).resolve().parent.parent
    script_path = root_dir / "deployment/create_script.sql"
    sdq_creator.write_create_sql_script(script_path)
    print("create_script written.")


if __name__ == "__main__":
    write_create_script()
