from pathlib import Path
from test.utils import postprocessing
from test.utils.parameters import model_params

import pytest

from exasol_transformers_extension.utils.bucketfs_model_specification import (
    BucketFSModelSpecification,
    get_BucketFSModelSpecification_from_model_Specs,
)

SUB_DIR = "test_delete_model_{id}"


@pytest.fixture()
def populate_models(setup_database, db_conn, bucketfs_location):
    def value_tuple_for_upload(
        model_spec: BucketFSModelSpecification,
    ) -> tuple[str, str, str, str, str]:
        return (
            model_spec.model_name,
            model_spec.task_type,
            str(model_spec.sub_dir),
            model_spec.bucketfs_conn_name,
            "",
        )

    bucketfs_conn_name, _ = setup_database
    n_rows = 5
    sub_dirs = []
    input_data: list[BucketFSModelSpecification] = []
    for i in range(n_rows):
        sub_dir = SUB_DIR.format(id=i)
        sub_dirs.append(sub_dir)
        current_model_specs = get_BucketFSModelSpecification_from_model_Specs(
            model_params.tiny_model_specs, bucketfs_conn_name, Path(sub_dir)
        )
        input_data.append(current_model_specs)

    try:
        query = f"""
            SELECT TE_MODEL_DOWNLOADER_UDF(
            t.model_name,
            t.task_type,
            t.sub_dir,
            t.bucketfs_conn_name,
            t.token_conn_name
            ) FROM (VALUES {str(tuple([value_tuple_for_upload(input_row) for input_row in input_data]))} AS
            t(model_name, task_type, sub_dir, bucketfs_conn_name, token_conn_name));
            """

        # execute downloader UDF
        db_conn.execute(query).fetchall()
        yield input_data

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


def astuple(model_spec: BucketFSModelSpecification) -> tuple[str, str, str, str]:
    return (
        model_spec.bucketfs_conn_name,
        str(model_spec.sub_dir),
        model_spec.model_name,
        model_spec.task_type,
    )


def test_delete_model(populate_models, db_conn, bucketfs_location):

    while len(populate_models) > 0:
        removed_model = populate_models.pop()
        query = f"""
            SELECT TE_DELETE_MODEL_UDF(
            t.bucketfs_conn_name,
            t.sub_dir,
            t.model_name,
            t.task_type
            ) FROM (VALUES {astuple(removed_model)} AS
            t(bucketfs_conn_name, sub_dir, model_name, task_type));
            """

        # execute downloader UDF
        res = db_conn.execute(query).fetchall()
        expected_res = [
            (
                removed_model.bucketfs_conn_name,
                str(removed_model.sub_dir),
                removed_model.model_name,
                removed_model.task_type,
                True,
                None,
            )
        ]
        assert res == expected_res

        for remaining_model in populate_models:
            validate_model_in_bucketfs(remaining_model, bucketfs_location)

        validate_model_not_in_bucketfs(removed_model, bucketfs_location)


def test_delete_model_error_wrong_bfs_conn(db_conn):
    model_location = BucketFSModelSpecification(
        model_name="not_existing_model",
        task_type="not_existing_task_type",
        sub_dir=Path("not_existing_sub_dir"),
        bucketfs_conn_name="not_existing_bucketfs_conn_name",
    )

    query = f"""
            SELECT TE_DELETE_MODEL_UDF(
            t.bucketfs_conn_name,
            t.sub_dir,
            t.model_name,
            t.task_type
            ) FROM (VALUES {astuple(model_location)} AS
            t(bucketfs_conn_name, sub_dir, model_name, task_type));
            """

    # execute downloader UDF
    res = db_conn.execute(query).fetchall()
    expected_res = (
        model_location.bucketfs_conn_name,
        str(model_location.sub_dir),
        model_location.model_name,
        model_location.task_type,
        False,
    )
    assert len(res) == 1, res
    assert res[0][:-1] == expected_res, res
    assert (
        f"get_connection for connection name {model_location.bucketfs_conn_name} failed: connection {model_location.bucketfs_conn_name.upper()} does not exist"
        in res[0][-1]
    )


def test_delete_model_error_wrong_model(db_conn, setup_database):
    bucketfs_conn_name, _ = setup_database

    model_location = BucketFSModelSpecification(
        model_name="not_existing_model",
        task_type="not_existing_task_type",
        sub_dir=Path("not_existing_sub_dir"),
        bucketfs_conn_name=bucketfs_conn_name,
    )

    query = f"""
            SELECT TE_DELETE_MODEL_UDF(
            t.bucketfs_conn_name,
            t.sub_dir,
            t.model_name,
            t.task_type
            ) FROM (VALUES {astuple(model_location)} AS
            t(bucketfs_conn_name, sub_dir, model_name, task_type));
            """

    # execute downloader UDF
    res = db_conn.execute(query).fetchall()
    expected_res = (
        model_location.bucketfs_conn_name,
        str(model_location.sub_dir),
        model_location.model_name,
        model_location.task_type,
        False,
    )
    assert len(res) == 1, res
    assert res[0][:-1] == expected_res, res
    assert (
        f"No such file or directory: 'container/{model_location.sub_dir}/{model_location.model_name}_{model_location.task_type}.tar.gz'"
        in res[0][-1]
    )
