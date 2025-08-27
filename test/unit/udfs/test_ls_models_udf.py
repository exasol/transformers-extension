import os

from test.unit.utils.utils_for_base_udf_tests import (
    create_mock_metadata,
)
from test.utils.parameters import model_params
from exasol_transformers_extension.udfs.models.ls_models_udf import (
    ListModelsUDF,
)
from pathlib import Path

from exasol_udf_mock_python.column import Column
from exasol_udf_mock_python.mock_meta_data import MockMetaData
from test.unit.utils.utils_for_udf_tests import (
    create_mock_exa_environment,
    create_mock_udf_context,
)
from test.utils.mock_connections import create_mounted_bucketfs_connection


import pytest

def create_mock_metadata():
    """Creates mock metadata for UDF tests"""
    meta = MockMetaData(
        script_code_wrapper_function=None,
        input_type="SET",
        input_columns=[
            Column("bucketfs_conn", str, "VARCHAR(2000000)"),
            Column("sub_dir", str, "VARCHAR(2000000)")
        ],
        output_type="EMITS",
        output_columns=[
            Column("bucketfs_conn", str, "VARCHAR(2000000)"),
            Column("sub_dir", str, "VARCHAR(2000000)"),
            Column("model_name", str, "VARCHAR(2000000)"),
            Column("task_name", str, "VARCHAR(2000000)"),
            Column("path", str, "VARCHAR(2000000)"),
            Column("error_message", str, "VARCHAR(2000000)"),
        ],
    )
    return meta


def test_ls_udf(tmpdir_factory):
    # get specs for a valid huggingface model

    token_model_specs = model_params.token_model_specs
    qa_model_specs = model_params.q_a_model_specs #todo these could be mocks
    sub_dir = "subdir"
    mock_bucketfs_location = tmpdir_factory.mktemp("test_list_models")
    # real bucketfs would create these dirs, but tempdir does not
    mock_bucketfs_location.mkdir(Path(sub_dir))
    mock_bucketfs_location.mkdir(Path(sub_dir) / "dslim")
    mock_bucketfs_location.mkdir(Path(sub_dir) / "deepset")#todo use made up modelspecs with different name formats
    mock_bucketfs_location.mkdir(Path(sub_dir) / token_model_specs.get_model_specific_path_suffix())
    mock_bucketfs_location.mkdir(Path(sub_dir) / qa_model_specs.get_model_specific_path_suffix())
    os.mknod(mock_bucketfs_location / sub_dir / token_model_specs.get_model_specific_path_suffix() + "/config.json")
    os.mknod(mock_bucketfs_location / sub_dir / qa_model_specs.get_model_specific_path_suffix() + "/renamed_config.json")
    for item in os.walk(mock_bucketfs_location):
        print(item)

    bucketfs_conn = create_mounted_bucketfs_connection(base_path=mock_bucketfs_location)

    bfs_conn_name = "bfs_conn"

    mock_meta = create_mock_metadata()
    mock_exa = create_mock_exa_environment(
        mock_meta, {bfs_conn_name: bucketfs_conn}
    )
    input_data = [(bfs_conn_name, sub_dir)]
    mock_ctx = create_mock_udf_context(input_data, mock_meta)

    udf = ListModelsUDF(
        exa=mock_exa
    )
    udf.run(mock_ctx)

    expected_output = [
        (bfs_conn_name, sub_dir, token_model_specs.model_name, token_model_specs.task_type,
         sub_dir / (token_model_specs.get_model_specific_path_suffix()), None),
        (bfs_conn_name, sub_dir, qa_model_specs.model_name, qa_model_specs.task_type,
         sub_dir / (qa_model_specs.get_model_specific_path_suffix()), None)
    ]
    print(mock_ctx.output)

    print("actual_path:", mock_ctx.output)
    print("expected_path:", expected_output)
    #assert (mock_bucketfs_location / expected_path[0]).exists()
    assert expected_output == mock_ctx.output




