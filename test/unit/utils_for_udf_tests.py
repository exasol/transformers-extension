"""Provides functions for creating mock exa-env and mock udf context"""
from typing import (
    Any,
    List,
    Tuple,
)

from exasol_udf_mock_python.connection import Connection
from exasol_udf_mock_python.mock_context import (
    StandaloneMockContext,
)
from exasol_udf_mock_python.mock_exa_environment import MockExaEnvironment
from exasol_udf_mock_python.mock_meta_data import MockMetaData


def create_mock_udf_context(
    input_data: List[Tuple[Any, ...]], mock_meta: MockMetaData
) -> StandaloneMockContext:
    """
    Creates mock udf context from given input data and mock meta data.
    returns mock_context object.
    """
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
    token_conn_obj: Connection,
) -> MockExaEnvironment:
    """
    Creates mock exa environment object form given mock metadata, bucketfs connection
    and token connection information.
    returns mock exa environment object
    """
    connections_dict = dict(zip(bfs_conn_name, bucketfs_connections))
    connections_dict[token_conn_name] = token_conn_obj
    mock_exa = MockExaEnvironment(metadata=mock_meta, connections=connections_dict)
    return mock_exa
