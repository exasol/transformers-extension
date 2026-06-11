from test.utils.db_queries import expected_script_list_all


def test_create_script(
    setup_database, db_conn
):
    """
    This test performs an installation of the extension.
    It then runs the model upload test to verify that the installation
    has brought the system into a ready-to-use state.
    """

    expected_scripts = expected_script_list_all

    with open("./create_script.sql", "w") as create_script:  # todo path
        query = create_script.read()

    result = db_conn.execute(query).fetchall()
    assert result.exit_code == 0

    list_scripts_query = """SELECT SCRIPT_NAME FROM EXA_ALL_SCRIPTS WHERE SCRIPT_LANGUAGE=PYTHON3_TE"""
    result = db_conn.execute(list_scripts_query).fetchall()

    assert result.exit_code == 0
    # verify all expected scripts are known by the database
    assert set(expected_scripts).issubset(set(result))