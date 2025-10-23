from __future__ import annotations

from pathlib import Path

import exasol.bucketfs as bfs


def cleanup_buckets(bucketfs_location: bfs.path.PathLike, path: str | Path):
    bucketfs_path = bucketfs_location / path
    if bucketfs_path.is_dir():
        bucketfs_path.rmdir(recursive=True)
    elif bucketfs_path.is_file():
        bucketfs_path.rm()
