from __future__ import annotations

from pathlib import Path
from typing import List

import exasol.bucketfs as bfs
from exasol_udf_mock_python.group import Group


def get_rounded_result(result: list[Group], round_: int = 2) -> list[tuple]:
    """
    Round the score value in each row, and re-creates the rows. Note that,
    `score` correspond to the 2nd column from the last. The `error_message`
    column is at the end of the lines.
    """

    ix_error_message = -1
    ix_score = -2

    rounded_result = result[0].rows
    for i in range(len(rounded_result)):
        row_result = ()
        row_result = rounded_result[i][:ix_score]
        if rounded_result[i][ix_score]:
            row_result += (round(rounded_result[i][ix_score], round_),)
        else:
            row_result += (rounded_result[i][ix_score],)
        row_result += (rounded_result[i][ix_error_message],)
        rounded_result[i] = row_result

    return rounded_result


def cleanup_buckets(bucketfs_location: bfs.path.PathLike, path: str | Path):
    bucketfs_path = bucketfs_location / path
    if bucketfs_path.is_dir():
        bucketfs_path.rmdir(recursive=True)
    elif bucketfs_path.is_file():
        bucketfs_path.rm()
