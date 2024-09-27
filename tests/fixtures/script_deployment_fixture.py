"""
This is rather ugly workaround for the problem with incompatible names of the DB and
BucketFS parameters, that are used in different scenarios.

In the ideal world, the parameters returned by the backend_aware_database_params and
backend_aware_bucketfs_params fixtures would be suitable for both creating respectively
a DB or BFS connection and using them in a command line (or more precisely, simulating
the command line in the context of tests). Unfortunately, the names and even the meanings
of some of the parameters in those two scenarios do not match.

At some point, we will standardise the names and replace the deploy_params and upload_params
fixtures with backend_aware_database_params and backend_aware_bucketfs_params.
"""
from __future__ import annotations
from typing import Any
from urllib.parse import urlparse
import pytest
from exasol.pytest_backend import BACKEND_ONPREM


_deploy_param_map = {
    'dsn': 'dsn',
    'user': 'db_user',
    'password': 'db_pass'
}

_upload_param_map = {
    'username': 'bucketfs-user',
    'password': 'bucketfs-password',
    'service_name': 'bucketfs-name',
    'bucket_name': 'bucket',
    'url': 'saas_url',
    'account_id': 'saas_account_id',
    'database_id': 'saas_database_id',
    'pat': 'saas_token'
}


def _parse_bucketfs_url(url: str) -> dict[str, Any]:
    parsed_url = urlparse(url)
    host, port = parsed_url.netloc.split(":")
    return {
        "bucketfs-host": host,
        "bucketfs-port": port,
        "bucketfs-use-https": parsed_url.scheme.lower() == 'https',
    }


def _translate_params(source: dict[str, Any], param_map: dict[str, str]) -> dict[str, Any]:
    return {param_map[k]: v for k, v in source.items() if k in param_map}


@pytest.fixture(scope="session")
def deploy_params(backend_aware_database_params) -> dict[str, Any]:
    return _translate_params(backend_aware_database_params, _deploy_param_map)


@pytest.fixture(scope="session")
def upload_params(backend, backend_aware_bucketfs_params) -> dict[str, Any]:
    if backend == BACKEND_ONPREM:
        mapped_params = _translate_params(backend_aware_bucketfs_params, _upload_param_map)
        mapped_params.pop(_upload_param_map['url'])
        mapped_params.update(_parse_bucketfs_url(backend_aware_bucketfs_params['url']))
        return mapped_params
    return _translate_params(backend_aware_bucketfs_params, _upload_param_map)
