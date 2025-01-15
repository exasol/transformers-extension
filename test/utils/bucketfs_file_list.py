from __future__ import annotations
from pathlib import PurePosixPath

import exasol.bucketfs as bfs

NODE_PATHLIKE_ID = 0
SUBDIR_LIST_ID = 1
FILE_LIST_ID = 2


def get_bucketfs_file_list(bucketfs_location: bfs.path.PathLike) -> list[str]:
    """
    Gets the list of files visible from a given BucketFS location. The returned file paths are
    relative to this location.
    """
    posix_location = PurePosixPath(str(bucketfs_location))
    return [str(PurePosixPath(str(node[NODE_PATHLIKE_ID] / leaf)).relative_to(posix_location))
            for node in bucketfs_location.walk()
            for leaf in node[FILE_LIST_ID] if not node[SUBDIR_LIST_ID]]
