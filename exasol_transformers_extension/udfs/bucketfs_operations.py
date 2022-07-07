from pathlib import PurePosixPath, Path
from exasol_bucketfs_utils_python import bucketfs_utils
from exasol_bucketfs_utils_python.bucketfs_factory import BucketFSFactory
from exasol_bucketfs_utils_python.bucketfs_location import BucketFSLocation


def create_bucketfs_location(bfs_conn_obj) -> BucketFSLocation:
    return BucketFSFactory().create_bucketfs_location(
        url=bfs_conn_obj.address,
        user=bfs_conn_obj.user,
        pwd=bfs_conn_obj.password)


def upload_model_files_to_bucketfs(
        tmpdir_name: str, model_path: str,
        bucketfs_location: BucketFSLocation) -> None:
    for tmp_file_path in Path(tmpdir_name).iterdir():
        with open(tmp_file_path, mode='rb') as file:
            bucketfs_path = PurePosixPath(
                model_path, tmp_file_path.relative_to(tmpdir_name))
            bucketfs_location.upload_fileobj_to_bucketfs(
                file, str(bucketfs_path))


def get_local_bucketfs_path(
        bucketfs_location: BucketFSLocation, model_path: str):
    # TODO: there is updated for unit test
    if bucketfs_location.__class__.__name__ == 'LocalFSMockBucketFSLocation':
        bucketfs_local_path = bucketfs_location.\
            get_complete_file_path_in_bucket(model_path)
    else:
        bucketfs_local_path = bucketfs_utils.generate_bucket_udf_path(
            bucketfs_location.bucket_config, model_path)
    return bucketfs_local_path


def get_model_path(model_name) -> str:
    return model_name.replace('-', '_')