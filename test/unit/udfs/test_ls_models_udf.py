import os
import pathlib
from unittest.mock import patch

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
    create_mock_udf_context, assert_result_matches_expected_output,
    assert_result_matches_expected_output_order_agnostic,
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

def setup_fake_model_files(mock_bucketfs_location, bfs_conn_name, sub_dir, token_model_specs, qa_model_specs):
    # real bucketfs would create these dirs, but tempdir does not
    mock_bucketfs_location.mkdir(Path(sub_dir))

    #these should not be found
    mock_bucketfs_location.mkdir(Path(sub_dir) / "not_a_model_dir")
    # outside sub_dir
    mock_bucketfs_location.mkdir("dslim")
    mock_bucketfs_location.mkdir(token_model_specs.get_model_specific_path_suffix())
    os.mknod(mock_bucketfs_location / token_model_specs.get_model_specific_path_suffix() + "/config.json")

    # these should be found
    mock_bucketfs_location.mkdir(Path(sub_dir) / "dslim")
    mock_bucketfs_location.mkdir(Path(sub_dir) / token_model_specs.get_model_specific_path_suffix())
    os.mknod(mock_bucketfs_location / sub_dir / token_model_specs.get_model_specific_path_suffix() + "/config.json")

    mock_bucketfs_location.mkdir(Path(sub_dir) / "deepset")
    mock_bucketfs_location.mkdir(Path(sub_dir) / qa_model_specs.get_model_specific_path_suffix())
    os.mknod(mock_bucketfs_location / sub_dir / qa_model_specs.get_model_specific_path_suffix() + "/config.json")

    mock_bucketfs_location.mkdir(Path(sub_dir) / "model_with_unknown_task_type")
    mock_bucketfs_location.mkdir(Path(sub_dir) / "model_with_unknown_task_type/model-name_unknown-task")
    os.mknod(mock_bucketfs_location / sub_dir / "model_with_unknown_task_type/model-name_unknown-task" + "/config.json")

    mock_bucketfs_location.mkdir(Path(sub_dir) / "model_with_no_task_type")
    mock_bucketfs_location.mkdir(Path(sub_dir) / "model_with_no_task_type/model-name-no-task")
    os.mknod(mock_bucketfs_location / sub_dir / "model_with_no_task_type/model-name-no-task" + "/config.json")

    expected_output = [
        (bfs_conn_name, sub_dir, token_model_specs.model_name, token_model_specs.task_type,
         str(mock_bucketfs_location /sub_dir / (token_model_specs.get_model_specific_path_suffix())), None),
        (bfs_conn_name, sub_dir, 'model_with_unknown_task_type/model-name', 'unknown-task',
         str(mock_bucketfs_location /sub_dir /'model_with_unknown_task_type/model-name_unknown-task'),
         "WARNING: We found a model which was saved using a task_name we don't recognize."),
        (bfs_conn_name, sub_dir, qa_model_specs.model_name, qa_model_specs.task_type,
         str(mock_bucketfs_location / sub_dir / (qa_model_specs.get_model_specific_path_suffix())), None),
        (bfs_conn_name, sub_dir, '', '',
         str(mock_bucketfs_location /sub_dir / 'model_with_no_task_type/model-name-no-task'),
         "ValueError: couldn't find a task name in path suffix model-name-no-task")
    ]
    return expected_output


def test_ls_udf(tmpdir_factory):
    # get specs for valid huggingface models
    token_model_specs = model_params.token_model_specs
    qa_model_specs = model_params.q_a_model_specs
    sub_dir = "subdir"
    mock_bucketfs_location = tmpdir_factory.mktemp("test_list_models")

    bfs_conn_name = "bfs_conn"
    bucketfs_conn = create_mounted_bucketfs_connection(base_path=mock_bucketfs_location)
    expected_output = setup_fake_model_files(mock_bucketfs_location, bfs_conn_name,
                                             sub_dir, token_model_specs, qa_model_specs)

    mock_meta = create_mock_metadata()
    mock_exa = create_mock_exa_environment(
        mock_meta, {bfs_conn_name: bucketfs_conn}
    )
    input_data = [(bfs_conn_name, sub_dir)]
    mock_ctx = create_mock_udf_context(input_data, mock_meta)

    with patch.object(ListModelsUDF, '_check_if_model_config', return_value=True) as _check_if_model_config:
        udf = ListModelsUDF(
            exa=mock_exa
        )
        udf.run(mock_ctx)
    assert_result_matches_expected_output_order_agnostic(mock_ctx.output, expected_output, ["bucketfs_conn", "sub_dir"], sort_by_column=4)



