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
            Column("model_name", str, "VARCHAR(2000000)"),
            Column("sub_dir", str, "VARCHAR(2000000)"),
            Column("task_type", str, "VARCHAR(2000000)"),
            Column("bfs_conn", str, "VARCHAR(2000000)"),
        ],
        output_type="EMITS",
        output_columns=[
            Column("model_name", str, "VARCHAR(2000000)"),
            Column("sub_dir", str, "VARCHAR(2000000)"),
            Column("task_type", str, "VARCHAR(2000000)"),
            Column("bfs_conn", str, "VARCHAR(2000000)"),
            Column("success", bool, "BOOLEAN"),
            Column("err_msg", str, "VARCHAR(2000000)"),
        ],
    )
    return meta


@pytest.mark.parametrize("count", list(range(1, 10)))
@patch(
    "exasol.python_extension_common.connections.bucketfs_location.create_bucketfs_location_from_conn_object"
)
@patch("exasol_transformers_extension.udfs.models.delete_model_udf.delete_model")
def test_delete_model(mock_delete_model, mock_create_loc, count):
    mock_bucketfs_locations = [Mock() for i in range(count)]
    mock_create_loc.side_effect = mock_bucketfs_locations
    base_model_names = [f"base_model_name_{i}" for i in range(count)]
    sub_directory_names = [f"sub_dir_{i}" for i in range(count)]
    task_type = [f"task_type_{i}" for i in range(count)]
    bucketfs_connections = [
        Connection(address=f"file:///test{i}") for i in range(count)
    ]
    bfs_conn_name = [f"bfs_conn_name_{i}" for i in bucketfs_connections]

    mock_current_model_specification_factory: Union[
        BucketFSModelSpecificationFactory, MagicMock
    ] = create_autospec(BucketFSModelSpecificationFactory)
    model_spec_mock = MagicMock(spec=BucketFSModelSpecification)
    mock_current_model_specification_factory.create.return_value = model_spec_mock

    input_data = [
        (
            base_model_names[i],
            sub_directory_names[i],
            task_type[i],
            bfs_conn_name[i],
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
        current_model_specification_factory=mock_current_model_specification_factory,
    )
    udf.run(mock_ctx)

    expected_bucketfs_model_specs_calls = [
        call(
            base_model_names[i],
            task_type[i],
            bfs_conn_name[i],
            Path(sub_directory_names[i]),
        )
        for i in range(count)
    ]
    assert (
        mock_current_model_specification_factory.create.mock_calls
        == expected_bucketfs_model_specs_calls
    )

    called_loc_addresses = [
        arg_list[0][0].address for arg_list in mock_create_loc.call_args_list
    ]
    expected_loc_addresses = [f"file:///test{i}" for i in range(count)]
    assert expected_loc_addresses == AnyOrder(called_loc_addresses)
    assert mock_ctx.output == [
        (
            base_model_names[i],
            task_type[i],
            sub_directory_names[i],
            bfs_conn_name[i],
            True,
            "",
        )
        for i in range(count)
    ]

    assert mock_delete_model.mock_calls == [
        call(mock_bucketfs_locations[i], model_spec_mock) for i in range(count)
    ]
