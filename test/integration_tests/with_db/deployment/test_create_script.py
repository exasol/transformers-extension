from test.utils.db_queries import expected_script_list_without_span

from exasol_transformers_extension.deployment.write_create_script import (
    write_create_script,
)


def test_create_script(setup_database, db_conn, tmpdir_factory):
    expected_scripts = expected_script_list_without_span

    # make sure we start out without scripts installed
    for script_name in expected_scripts:
        db_conn.execute(f"DROP SCRIPT {script_name};")

    tmpdir = tmpdir_factory.mktemp("test_create_script")
    script_path = write_create_script(root_dir=tmpdir)

    with open(script_path) as create_script:
        queries = create_script.read()

    query_list = queries.split("-- next call:\n\n")

    for query in query_list[:-1]:
        db_conn.execute(query)

    list_scripts_query = """SELECT SCRIPT_NAME FROM EXA_ALL_SCRIPTS"""
    result = db_conn.execute(list_scripts_query).fetchall()
    found_scripts = [x[0] for x in result]

    # verify all expected scripts are known by the database
    assert set(expected_scripts).issubset(set(found_scripts))
