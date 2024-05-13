import subprocess
import tarfile
import tempfile
from pathlib import PurePosixPath, Path
from typing import BinaryIO

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
        bucketfs_location: AbstractBucketFSLocation) -> Path:
    """
    uploads model in tmpdir_name to model_path in bucketfs_location
    """
    with tempfile.TemporaryFile() as fileobj:
        create_tar_of_directory(Path(tmpdir_name), fileobj)
        model_upload_tar_file_path = model_path.with_suffix(".tar.gz")
        return upload_file_to_bucketfs_with_retry(bucketfs_location, fileobj, model_upload_tar_file_path)


@retry(wait=wait_fixed(2), stop=stop_after_attempt(10))
def upload_file_to_bucketfs_with_retry(bucketfs_location: AbstractBucketFSLocation,
                                       fileobj: BinaryIO,
                                       file_path: Path) -> Path:
    fileobj.seek(0)
    bucketfs_location.upload_fileobj_to_bucketfs(fileobj, str(file_path))
    return file_path


def create_tar_of_directory(path: Path, fileobj: BinaryIO):
    with tarfile.open(name="model.tar.gz", mode="w|gz", fileobj=fileobj) as tar:
        for subpath in path.glob("*"):
            tar.add(name=subpath, arcname=subpath.name)


def get_local_bucketfs_path(
        bucketfs_location: BucketFSLocation, model_path: str) -> PurePosixPath:
    bucketfs_local_path = bucketfs_location.generate_bucket_udf_path(model_path)
    return bucketfs_local_path

# path used to save model in "local bucketfs" by prepare_model_for_local_bucketfs fixtures for testing
# todo this should probably better reflect create_save_pretrained_model_path
def generate_local_bucketfs_path_for_model(tmpdir: Path, sub_dir, model: str):
    return tmpdir / sub_dir / model

# path used to save model in bucketfs by upload_*_model_to_bucketfs fixtures for testing
# todo this should probably better reflect create_save_pretrained_model_path
def get_model_path(sub_dir: str, model_name: str) -> Path:
    return Path(sub_dir, model_name)

# path model is saved at in the bucketfs
def get_model_path_with_pretrained(sub_dir: str, model_name: str) -> Path:
    return Path(sub_dir, model_name, "pretrained", model_name)


# path HuggingFaceHubBucketFSModelTransferSP saves the model at using sae_pretrained,
# befor it is uploaded to the bucketfs
def create_save_pretrained_model_path(_tmpdir_name, _model_name) -> Path: #todo here we will add the version and task to the path later
    return Path(_tmpdir_name, "pretrained", _model_name)


