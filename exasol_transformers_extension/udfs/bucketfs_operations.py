from pathlib import PurePosixPath, Path
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


def download_model_files_from_bucketfs(
        tmpdir_name: str, model_path,
        bucketfs_location: BucketFSLocation) -> None:
    model_files = bucketfs_location.list_files_in_bucketfs(model_path)
    for model_file in model_files:
        bucketfs_location.read_file_from_bucketfs_to_file(
            bucket_file_path=str(Path(model_path, model_file)),
            local_file_path=Path(tmpdir_name, model_file))


def get_model_path(model_name) -> str:
    return model_name.replace('-', '_')
