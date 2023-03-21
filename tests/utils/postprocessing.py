from pathlib import PurePosixPath
from typing import List
from exasol_bucketfs_utils_python.abstract_bucketfs_location import \
    AbstractBucketFSLocation
from exasol_udf_mock_python.group import Group


def get_rounded_result(result: List[Group], round_: int = 2) -> List[tuple]:
    rounded_result = result[0].rows
    for i in range(len(rounded_result)):
        rounded_result[i] = rounded_result[i][:-2] + \
                            (round(rounded_result[i][-2], round_),) + \
                            (rounded_result[i][-1], )
    return rounded_result


def cleanup_buckets(bucketfs_location: AbstractBucketFSLocation, path: str):
    bucketfs_files = bucketfs_location.list_files_in_bucketfs(path)
    for file_ in bucketfs_files:
        try:
            bucketfs_location.delete_file_in_bucketfs(
                str(PurePosixPath(path, file_)))
        except Exception as exc:
            print(f"Error while deleting downloaded files, {str(exc)}")

