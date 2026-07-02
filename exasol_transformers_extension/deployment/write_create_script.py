"""
Writes SQL queries for the creation of all udf script to create_script.sql
"""

import os
from pathlib import Path

from pygments.lexers import sql

from exasol_transformers_extension.deployment.script_deployment_queries import (
    ScriptDeploymentQueries,
)


def write_create_script(root_dir=None) -> Path:
    """
    Writes SQL queries for the creation of all udf script to
    <root_dir>/"deployment/create_script.sql"create_script.sql
    """
    print("Write create script")
    sql_creator = ScriptDeploymentQueries(
        language_alias="PYTHON3_TE", use_spans=False, install_all_scripts=False
    )

    if not root_dir:
        root_dir = Path(__file__).resolve().parent.parent

    dir_path = root_dir / "deployment"

    if not os.path.exists(dir_path):  # create folders if not exists
        os.makedirs(dir_path)
    script_path = dir_path / "create_script.sql"
    sql_creator.write_create_sql_script(script_path)
    print("create_script written.")
    return script_path


if __name__ == "__main__":
    write_create_script(None)
