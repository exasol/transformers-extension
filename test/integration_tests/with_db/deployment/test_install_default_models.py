from test.integration_tests.with_db.udfs.python_rows_to_sql import python_rows_to_sql
from test.utils import postprocessing

from click.testing import CliRunner
from exasol.python_extension_common.cli.std_options import (
    StdParams,
    get_cli_arg,
)

from exasol_transformers_extension.deployment.default_udf_parameters import (
    DEFAULT_VALUES,
)
from exasol_transformers_extension.install_default_models import (
    install_default_models_command,
)


def test_install_default_models_cli(
    bucketfs_cli_args,
    setup_database,
    db_conn,
    bucketfs_location,
):

    args_string = " ".join(
        [
            bucketfs_cli_args,
            get_cli_arg(StdParams.path_in_bucket, str(bucketfs_location)),
        ]
    )

    runner = CliRunner()
    result = runner.invoke(
        install_default_models_command, args=args_string, catch_exceptions=False
    )
    if result.exit_code != 0:
        print("Exception:", result.exception)
        print("ExcInfo:", result.exc_info)
        print("STDERR:", result.stderr_bytes)
        print("STDOUT:", result.stdout_bytes)
    assert result.exit_code == 0

    try:
        sentiment_text = "I am so happy to be working on the Transformers Extension."
        input_data = [(sentiment_text,)]

        query = (
            f"SELECT AI_SENTIMENT("
            f"t.text_data,"
            f") FROM (VALUES {python_rows_to_sql(input_data)} "
            f"AS t(text_data));"
        )

        result = db_conn.execute(query).fetchall()
        assert len(result) == 1 and result[0][-1] is None

        extract_text = "The database software company Exasol is based in Nuremberg"
        input_data = [(extract_text,)]

        query = (
            f"SELECT  AI_EXTRACT_ENTITIES("
            f"t.text_data,"
            f") FROM (VALUES {python_rows_to_sql(input_data)} "
            f"AS t(text_data));"
        )

        result = db_conn.execute(query).fetchall()
        assert len(result) == 1 and result[0][-1] is None

        candidate_labels = "Database,Analytics,Germany,Food,Party"
        class_text = "The database software company Exasol is based in Nuremberg"

        input_data = [(class_text, candidate_labels)]

        query = (
            f"SELECT AI_CLASSIFY("
            f"t.text_data,"
            f"t.candidate_labels,"
            f") FROM (VALUES {python_rows_to_sql(input_data)} "
            f"AS t(text_data, candidate_labels));"
        )

        result = db_conn.execute(query).fetchall()
        assert len(result) == 1 and result[0][-1] is None
    finally:
        postprocessing.cleanup_buckets(bucketfs_location, DEFAULT_VALUES.sub_dir)
