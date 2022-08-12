from pathlib import PurePosixPath, Path
from exasol_bucketfs_utils_python.abstract_bucketfs_location import \
    AbstractBucketFSLocation
from exasol_bucketfs_utils_python.bucketfs_factory import BucketFSFactory
from exasol_bucketfs_utils_python.bucketfs_location import BucketFSLocation


def create_bucketfs_location(bfs_conn_obj) -> BucketFSLocation:
    return BucketFSFactory().create_bucketfs_location(
        url=bfs_conn_obj.address,
        user=bfs_conn_obj.user,
        pwd=bfs_conn_obj.password)


def upload_model_files_to_bucketfs(
        tmpdir_name: str, model_path: Path,
        bucketfs_location: AbstractBucketFSLocation) -> None:
    for tmp_file_path in Path(tmpdir_name).iterdir():
        with open(tmp_file_path, mode='rb') as file:
            bucketfs_path = PurePosixPath(
                model_path, tmp_file_path.relative_to(tmpdir_name))
            bucketfs_location.upload_fileobj_to_bucketfs(
                file, str(bucketfs_path))


def get_local_bucketfs_path(
        bucketfs_location: BucketFSLocation, model_path: str) -> PurePosixPath:
    bucketfs_local_path = bucketfs_location.generate_bucket_udf_path(model_path)
    return bucketfs_local_path


def get_model_path(sub_dir: str, model_name: str) -> Path:
    return Path(sub_dir, model_name.replace('-', '_'))
