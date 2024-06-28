from pathlib import Path, PosixPath

from click.testing import CliRunner
import exasol.bucketfs as bfs

from exasol_transformers_extension import upload_model
from exasol_transformers_extension.utils.current_model_specification import \
    CurrentModelSpecificationFromModelSpecs
from tests.integration_tests.with_db.udfs.python_rows_to_sql import python_rows_to_sql
from tests.utils import postprocessing
from tests.utils.parameters import bucketfs_params, model_params, get_arg_list
from tests.fixtures.model_fixture import download_model_to_standard_local_save_path


def adapt_file_to_upload(path: PosixPath, download_path: PosixPath):
    if path.is_dir():
        path = path / "not_empty"
    if ".no_exist" in path.parts:
        parts = list(path.parts)
        parts[path.parts.index(".no_exist")] = "no_exist"
        path = PosixPath(*parts)
    path = path.relative_to(download_path)
    return PosixPath(path)


def test_model_upload(upload_params,
                      setup_database,
                      pyexasol_connection,
                      tmp_path: Path,
                      bucketfs_location: bfs.path.PathLike):

    # Debugging
    assert pyexasol_connection.execute(f"SELECT CURRENT_SCHEMA;").fetchval() == setup_database[1]

    sub_dir = 'sub_dir'
    model_specification = model_params.base_model_specs
    model_name = model_specification.model_name
    download_path = download_model_to_standard_local_save_path(model_specification, tmp_path)
    current_model_specs = CurrentModelSpecificationFromModelSpecs().transform(model_specification,
                                                                              "", Path(sub_dir))
    upload_path = current_model_specs.get_bucketfs_model_save_path()
    args_list = get_arg_list(**upload_params,
                             path_in_bucket=bucketfs_params.path_in_bucket,
                             model_name=model_name,
                             sub_dir=sub_dir,
                             local_model_path=str(download_path))

    try:
        runner = CliRunner()
        result = runner.invoke(upload_model.main, args_list)
        assert result.exit_code == 0
        bucketfs_upload_location = bucketfs_location / upload_path.with_suffix(".tar.gz")
        assert bucketfs_upload_location.is_file()

        bucketfs_conn_name, schema_name = setup_database
        text_data = "Exasol is an analytics <mask> management software company."
        input_data = [
            (
                '',
                bucketfs_conn_name,
                sub_dir,
                model_name,
                text_data,
                1
            )
        ]

        query = f"SELECT TE_FILLING_MASK_UDF(" \
                f"t.device_id, " \
                f"t.bucketfs_conn_name, " \
                f"t.sub_dir, " \
                f"t.model_name, " \
                f"t.text_data," \
                f"t.top_k" \
                f") FROM (VALUES {python_rows_to_sql(input_data)} " \
                f"AS t(device_id, bucketfs_conn_name, sub_dir, " \
                f"model_name, text_data, top_k));"

        # execute sequence classification UDF
        result = pyexasol_connection.execute(query).fetchall()
        assert len(result) == 1 and result[0][-1] is None
    finally:
        postprocessing.cleanup_buckets(bucketfs_location, sub_dir)

        # Debugging
        assert pyexasol_connection.execute(f"SELECT CURRENT_SCHEMA;").fetchval() == setup_database[1]
