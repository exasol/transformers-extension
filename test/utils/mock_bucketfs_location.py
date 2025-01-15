from pathlib import PurePosixPath


def fake_bucketfs_location_from_conn_object(bfs_conn_obj):
    return PurePosixPath(bfs_conn_obj.address[7:])


def fake_local_bucketfs_path(bucketfs_location, model_path):
    return bucketfs_location / model_path
