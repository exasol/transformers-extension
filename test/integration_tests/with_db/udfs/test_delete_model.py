from pathlib import Path
from test.integration_tests.with_db.udfs.python_rows_to_sql import (
    python_row_to_sql,
    python_rows_to_sql,
)
from test.utils import postprocessing
from test.utils.parameters import model_params

import pytest

from exasol_transformers_extension.utils.bucketfs_model_specification import (
    BucketFSModelSpecification,
)

SUB_DIR = "test_delete_model_{id}"


def model_spec_as_input_tuple(
    model_spec: BucketFSModelSpecification,
) -> tuple[str, str, str, str]:
    """returns tuple with model spec values in right order to be used as udf input"""
    return (
        model_spec.bucketfs_conn_name,
        str(model_spec.sub_dir),
        model_spec.model_name,
        model_spec.task_type,
    )


@pytest.fixture()
def populate_models(setup_database, db_conn, bucketfs_location):

    bucketfs_conn_name, _ = setup_database
    n_rows = 5
    sub_dirs = []
    input_data_model_specs = []
    for i in range(n_rows):
        sub_dir = SUB_DIR.format(id=i)
        sub_dirs.append(sub_dir)
        input_data_model_specs.append(
            BucketFSModelSpecification(
                bucketfs_conn_name=bucketfs_conn_name,
                sub_dir=Path(sub_dir),
                model_name=model_params.tiny_model_specs.model_name,
                task_type=model_params.tiny_model_specs.task_type,
            )
        )
    try:
        query = f"""
            SELECT TE_MODEL_DOWNLOADER_UDF(
            t.bucketfs_conn_name,
            t.sub_dir,
            t.model_name,
            t.task_type,
            t.token_conn_name
            ) FROM (VALUES {python_rows_to_sql(
                                [ model_spec_as_input_tuple(model) + ("",) for model in input_data_model_specs] 
                            )} AS
            t(bucketfs_conn_name, sub_dir, model_name, task_type, token_conn_name));
            """

        # execute downloader UDF
        db_conn.execute(query).fetchall()
        yield input_data_model_specs

    finally:
        for sub_dir in sub_dirs:
            postprocessing.cleanup_buckets(bucketfs_location, sub_dir)


def _build_path(model_location: BucketFSModelSpecification, bucketfs_location):
    p = bucketfs_location / model_location.get_bucketfs_model_save_path().with_suffix(
        ".tar.gz"
    )
    return p


def validate_model_in_bucketfs(
    model_location: BucketFSModelSpecification, bucketfs_location
):
    assert _build_path(model_location, bucketfs_location).exists()


def validate_model_not_in_bucketfs(
    model_location: BucketFSModelSpecification, bucketfs_location
):
    p = _build_path(model_location, bucketfs_location)
    assert not p.exists(), f"Path {p} supposed to be deleted, but still exists"


def test_delete_model(populate_models, db_conn, bucketfs_location):

    while len(populate_models) > 0:
        removed_model_spec = populate_models.pop()
        removed_model_tuple = model_spec_as_input_tuple(removed_model_spec)
        query = f"""
            SELECT TE_DELETE_MODEL_UDF(
            t.bucketfs_conn_name,
            t.sub_dir,
            t.model_name,
            t.task_type
            ) FROM (VALUES {python_row_to_sql(removed_model_tuple)} AS
            t(bucketfs_conn_name, sub_dir, model_name, task_type));
            """

        # execute downloader UDF
        res = db_conn.execute(query).fetchall()
        expected_res = [
            removed_model_tuple
            + (
                True,
                None,
            )
        ]
        assert res == expected_res

        for remaining_model in populate_models:
            validate_model_in_bucketfs(remaining_model, bucketfs_location)

        validate_model_not_in_bucketfs(removed_model_spec, bucketfs_location)


def run_delete_model_error_test(
    db_connection,
    model_location_spec: BucketFSModelSpecification,
    expected_error_message: str,
):
    model_location_tuple = model_spec_as_input_tuple(model_location_spec)

    query = f"""
            SELECT TE_DELETE_MODEL_UDF(
            t.bucketfs_conn_name,
            t.sub_dir,
            t.model_name,
            t.task_type
            ) FROM (VALUES {python_row_to_sql(model_location_tuple)} AS
            t(bucketfs_conn_name, sub_dir, model_name, task_type));
            """

    # execute downloader UDF
    res = db_connection.execute(query).fetchall()
    expected_res = model_location_tuple + (False,)

    assert len(res) == 1, res
    assert res[0][:-1] == expected_res, res
    assert expected_error_message in res[0][-1]


def test_delete_model_error_wrong_bfs_conn(db_conn):
    model_location_spec = BucketFSModelSpecification(
        model_name="not_existing_model",
        task_type="fill_mask",
        sub_dir=Path("not_existing_sub_dir"),
        bucketfs_conn_name="not_existing_bucketfs_conn_name",
    )
    model_location_spec.task_type = (
        model_location_spec.legacy_set_task_type_from_udf_name("not_existing_task_type")
    )
    expected_error_message = (
        f"get_connection for connection name {model_location_spec.bucketfs_conn_name} "
        f"failed: connection {model_location_spec.bucketfs_conn_name.upper()} does "
        f"not exist"
    )
    run_delete_model_error_test(db_conn, model_location_spec, expected_error_message)


def test_delete_model_error_wrong_model(db_conn, setup_database):
    bucketfs_conn_name, _ = setup_database

    model_location_spec = BucketFSModelSpecification(
        bucketfs_conn_name=bucketfs_conn_name,
        sub_dir=Path("not_existing_sub_dir"),
        model_name="not_existing_model",
        task_type="fill_mask",
    )
    model_location_spec.task_type = (
        model_location_spec.legacy_set_task_type_from_udf_name("not_existing_task_type")
    )
    expected_error_message = (
        f"No such file or directory: 'container/{model_location_spec.sub_dir}/"
        f"{model_location_spec.model_name}_{model_location_spec.task_type}.tar.gz'"
    )
    run_delete_model_error_test(db_conn, model_location_spec, expected_error_message)
