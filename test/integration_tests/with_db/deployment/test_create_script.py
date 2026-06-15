from pathlib import Path
from test.utils.db_queries import expected_script_list_all

from exasol_transformers_extension.deployment.write_create_script import (
    write_create_script,
)


def test_create_script(setup_database, db_conn, tmpdir_factory):
    expected_scripts = expected_script_list_all

    # make sure we start out without scripts installed
    for script_name in expected_scripts:
        db_conn.execute(f"DROP SCRIPT {script_name};")

    tmpdir = tmpdir_factory.mktemp("test_create_script")
    script_path = write_create_script(root_dir=tmpdir)

    with open(script_path, "r") as create_script:
        query = create_script.read()

    db_conn.execute(query)

    list_scripts_query = (
        """SELECT SCRIPT_NAME FROM EXA_ALL_SCRIPTS"""
    )
    result = db_conn.execute(list_scripts_query).fetchall()

    assert result.exit_code == 0
    # verify all expected scripts are known by the database
    assert set(expected_scripts).issubset(set(result))
