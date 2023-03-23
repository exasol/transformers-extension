from pathlib import PurePosixPath
from typing import List
from exasol_bucketfs_utils_python.abstract_bucketfs_location import \
    AbstractBucketFSLocation
from exasol_udf_mock_python.group import Group


def get_rounded_result(result: List[Group], round_: int = 2) -> List[tuple]:
    """
    Round the score value in each row, and re-creates the rows. Note that,
    `score` correspond to the 2nd column from the last. The `error_message`
    column is at the end of the lines.
    """

    ix_error_message = -1
    ix_score = -2

    rounded_result = result[0].rows
    for i in range(len(rounded_result)):
        rounded_result[i] = rounded_result[i][:ix_score] + \
                            (round(rounded_result[i][ix_score], round_),) + \
                            (rounded_result[i][ix_error_message], )

    return rounded_result


def cleanup_buckets(bucketfs_location: AbstractBucketFSLocation, path: str):
    bucketfs_files = bucketfs_location.list_files_in_bucketfs(path)
    for file_ in bucketfs_files:
        try:
            bucketfs_location.delete_file_in_bucketfs(
                str(PurePosixPath(path, file_)))
        except Exception as exc:
            print(f"Error while deleting downloaded files, {str(exc)}")

