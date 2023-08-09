from pathlib import PurePosixPath, Path

from exasol_bucketfs_utils_python.abstract_bucketfs_location import \
    AbstractBucketFSLocation
from exasol_bucketfs_utils_python.bucket_config import BucketConfig
from exasol_bucketfs_utils_python.bucketfs_config import BucketFSConfig
from exasol_bucketfs_utils_python.bucketfs_connection_config import \
    BucketFSConnectionConfig
from exasol_bucketfs_utils_python.bucketfs_factory import BucketFSFactory
from exasol_bucketfs_utils_python.bucketfs_location import BucketFSLocation
from tenacity import retry, wait_fixed, stop_after_attempt


def create_bucketfs_location_from_conn_object(bfs_conn_obj) -> BucketFSLocation:
    return BucketFSFactory().create_bucketfs_location(
        url=bfs_conn_obj.address,
        user=bfs_conn_obj.user,
        pwd=bfs_conn_obj.password)


def create_bucketfs_location(
        bucketfs_name: str, bucketfs_host: str, bucketfs_port: int,
        bucketfs_use_https: bool, bucketfs_user: str, bucketfs_password: str,
        bucket: str, path_in_bucket: str) -> BucketFSLocation:
    _bucketfs_connection = BucketFSConnectionConfig(
        host=bucketfs_host, port=bucketfs_port, user=bucketfs_user,
        pwd=bucketfs_password, is_https=bucketfs_use_https)
    _bucketfs_config = BucketFSConfig(
        bucketfs_name=bucketfs_name, connection_config=_bucketfs_connection)
    _bucket_config = BucketConfig(
        bucket_name=bucket, bucketfs_config=_bucketfs_config)
    return BucketFSLocation(
        bucket_config=_bucket_config,
        base_path=PurePosixPath(path_in_bucket))


def upload_model_files_to_bucketfs(
        tmpdir_name: str, model_path: Path,
        bucketfs_location: AbstractBucketFSLocation) -> None:
    for path in Path(tmpdir_name).rglob("*"):
        relative_path = path.relative_to(tmpdir_name)
        if path.is_file():
            upload_file(bucketfs_location, model_path, path, relative_path)
        elif path.is_dir():
            upload_not_empty_for_directory(bucketfs_location, model_path, relative_path)


@retry(wait=wait_fixed(2), stop=stop_after_attempt(10))
def upload_file(bucketfs_location: AbstractBucketFSLocation, model_path: Path, path: Path, relative_path: Path):
    with open(path, mode='rb') as file:
        bucketfs_path = PurePosixPath(model_path, relative_path)
        bucketfs_location.upload_fileobj_to_bucketfs(file, str(bucketfs_path))


@retry(wait=wait_fixed(2), stop=stop_after_attempt(10))
def upload_not_empty_for_directory(bucketfs_location: AbstractBucketFSLocation, model_path: Path, relative_path: Path):
    bucketfs_path = PurePosixPath(model_path, relative_path, ".not_empty")
    bucketfs_location.upload_string_to_bucketfs(str(bucketfs_path), "")


def get_local_bucketfs_path(
        bucketfs_location: BucketFSLocation, model_path: str) -> PurePosixPath:
    bucketfs_local_path = bucketfs_location.generate_bucket_udf_path(model_path)
    return bucketfs_local_path


def get_model_path(sub_dir: str, model_name: str) -> Path:
    return Path(sub_dir, model_name.replace('-', '_'))
