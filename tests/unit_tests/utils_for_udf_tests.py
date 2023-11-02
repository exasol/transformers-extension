from typing import Any, Tuple, List

from exasol_udf_mock_python.mock_context import StandaloneMockContext, MockContext
from exasol_udf_mock_python.connection import Connection
from exasol_udf_mock_python.mock_exa_environment import MockExaEnvironment
from exasol_udf_mock_python.mock_meta_data import MockMetaData


def create_mock_udf_context(input_data: List[Tuple[Any, ...]], mock_meta: MockMetaData) -> StandaloneMockContext:
    mock_ctx = StandaloneMockContext(
        inp=input_data,
        metadata=mock_meta,
    )
    return mock_ctx


def create_mock_exa_environment(
        bfs_conn_name: List[str],
        bucketfs_connections: List[Connection],
        mock_meta: MockMetaData,
        token_conn_name: str,
        token_conn_obj: Connection) -> MockExaEnvironment:
    connections_dict = {k: v for k, v in zip(bfs_conn_name, bucketfs_connections)}
    connections_dict[token_conn_name] = token_conn_obj
    mock_exa = MockExaEnvironment(
        metadata=mock_meta,
        connections=connections_dict
    )
    return mock_exa