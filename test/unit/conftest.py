import os

import pytest


@pytest.fixture(autouse=True)
def mock_virtual_bucketfs_paths(monkeypatch):
    real_exists = os.path.exists

    def exists(path):
        path_string = os.fspath(path)
        return path_string.startswith(
            ("/tmpdir_", "/test/", "test/Path")
        ) or real_exists(path)

    monkeypatch.setattr(os.path, "exists", exists)
