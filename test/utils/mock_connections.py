from __future__ import annotations

from pathlib import (
    Path,
    PurePosixPath,
)

import exasol.bucketfs as bfs
from exasol_udf_mock_python.connection import Connection


def create_mounted_bucketfs_connection(
    base_path: Path | PurePosixPath | str, path_in_bucket: PurePosixPath | str = ""
) -> Connection:
    address = (
        f'{{"backend":"{bfs.path.StorageBackend.mounted.name}", '
        f'"base_path":"{base_path}", "path":"{path_in_bucket}"}}'
    )
    return Connection(address=address, user="{}", password="{}")


def create_hf_token_connection(token: str) -> Connection:
    return Connection(address="", user="", password=token)
