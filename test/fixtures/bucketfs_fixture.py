import exasol.bucketfs as bfs
import pytest
from test.utils.parameters import PATH_IN_BUCKET


@pytest.fixture(scope="session")
def bucketfs_location(backend_aware_bucketfs_params) -> bfs.path.PathLike:
    return bfs.path.build_path(**backend_aware_bucketfs_params,
                               path=PATH_IN_BUCKET)
