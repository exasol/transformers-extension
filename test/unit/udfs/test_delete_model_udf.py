from pathlib import Path
from test.unit.utils.utils_for_udf_tests import (
    create_mock_exa_environment,
    create_mock_udf_context,
)
from test.utils.matchers import AnyOrder
from typing import (
    Union,
)
from unittest.mock import (
    MagicMock,
    Mock,
    call,
    create_autospec,
    patch,
)

import pytest
from _pytest.monkeypatch import MonkeyPatch
from exasol_udf_mock_python.column import Column
from exasol_udf_mock_python.connection import Connection
from exasol_udf_mock_python.mock_meta_data import MockMetaData

from exasol_transformers_extension.udfs.models.delete_model_udf import DeleteModelUDF
from exasol_transformers_extension.utils.bucketfs_model_specification import (
    BucketFSModelSpecification,
    BucketFSModelSpecificationFactory,
)


def create_mock_metadata() -> MockMetaData:
    meta = MockMetaData(
        script_code_wrapper_function=None,
        input_type="SET",
        input_columns=[
            Column("bucketfs_conn", str, "VARCHAR(2000000)"),
            Column("sub_dir", str, "VARCHAR(2000000)"),
            Column("model_name", str, "VARCHAR(2000000)"),
            Column("task_type", str, "VARCHAR(2000000)"),
        ],
        output_type="EMITS",
        output_columns=[
            Column("bucketfs_conn", str, "VARCHAR(2000000)"),
            Column("sub_dir", str, "VARCHAR(2000000)"),
            Column("model_name", str, "VARCHAR(2000000)"),
            Column("task_type", str, "VARCHAR(2000000)"),
            Column("success", bool, "BOOLEAN"),
            Column("error_message", str, "VARCHAR(2000000)"),
        ],
    )
    return meta


@pytest.mark.parametrize("count", list(range(1, 10)))
@pytest.mark.parametrize(
    "task_type", ["fill-mask", "illegal-task-type", "sequence-classification"]
)
@patch(
    "exasol.python_extension_common.connections.bucketfs_location.create_bucketfs_location_from_conn_object"
)
@patch("exasol_transformers_extension.udfs.models.delete_model_udf.delete_model")
def test_delete_model(mock_delete_model, mock_create_loc, count, task_type):

    mock_bucketfs_locations = [Mock() for i in range(count)]
    mock_create_loc.side_effect = mock_bucketfs_locations
    base_model_names = [f"base_model_name_{i}" for i in range(count)]
    sub_directory_names = [f"sub_dir_{i}" for i in range(count)]
    bucketfs_connections = [
        Connection(address=f"file:///test{i}") for i in range(count)
    ]
    bfs_conn_name = [f"bfs_conn_name_{i}" for i in bucketfs_connections]

    input_data = [
        (
            bfs_conn_name[i],
            sub_directory_names[i],
            base_model_names[i],
            task_type,
        )
        for i in range(count)
    ]
    mock_meta = create_mock_metadata()
    mock_exa = create_mock_exa_environment(
        mock_meta, dict(zip(bfs_conn_name, bucketfs_connections))
    )
    mock_ctx = create_mock_udf_context(input_data, mock_meta)

    udf = DeleteModelUDF(
        exa=mock_exa,
    )
    udf.run(mock_ctx)

    def create_bfs_model_spec_with_any_task_type(
        model_name, task_type, bfs_conn_name, subdir
    ):
        model_spec = BucketFSModelSpecification(
            model_name, "fill_mask", bfs_conn_name, subdir
        )
        model_spec.task_type = model_spec.legacy_set_task_type_from_udf_name(task_type)
        return model_spec

    expected_bucketfs_model_specs = [
        create_bfs_model_spec_with_any_task_type(
            base_model_names[i],
            task_type,
            bfs_conn_name[i],
            Path(sub_directory_names[i]),
        )
        for i in range(count)
    ]

    called_loc_addresses = [
        arg_list[0][0].address for arg_list in mock_create_loc.call_args_list
    ]
    expected_loc_addresses = [f"file:///test{i}" for i in range(count)]
    assert expected_loc_addresses == AnyOrder(called_loc_addresses)
    assert mock_ctx.output == [
        (
            bfs_conn_name[i],
            sub_directory_names[i],
            base_model_names[i],
            task_type,
            True,
            "",
        )
        for i in range(count)
    ]

    assert mock_delete_model.mock_calls == [
        call(mock_bucketfs_locations[i], expected_bucketfs_model_specs[i])
        for i in range(count)
    ]
