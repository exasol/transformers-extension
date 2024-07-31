from exasol_transformers_extension.utils.model_specification import ModelSpecification
from pathlib import PurePosixPath, Path

class BucketFSModelSpecification(ModelSpecification):
    """
    Class describing a model with additional information about
    the bucketFS connection and the subdir in the bucketfs the model can be found at.
    """
    def __init__(self,
                 model_name: str,
                 task_type: str,
                 bucketfs_conn_name: str,
                 sub_dir: Path):
        ModelSpecification.__init__(self, model_name, task_type)
        self.bucketfs_conn_name = bucketfs_conn_name
        self.sub_dir = sub_dir

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, BucketFSModelSpecification):
            return (super().__eq__(other)
                    and self.sub_dir == other.sub_dir and
                    self.bucketfs_conn_name == other.bucketfs_conn_name)
        return False

    def get_bucketfs_model_save_path(self) -> Path:
        """
        path model is saved at in the bucketfs
        """
        model_path_suffix = self.get_model_specific_path_suffix()
        return Path(self.sub_dir, model_path_suffix)


class BucketFSModelSpecificationFactory:
    def create(self,
               model_name: str,
               task_type: str,
               bucketfs_conn_name: str,
               sub_dir: Path):
        return BucketFSModelSpecification(model_name, task_type, bucketfs_conn_name, sub_dir)


def get_BucketFSModelSpecification_from_model_Specs(
        model_specification: ModelSpecification,
        bucketfs_conn_name: str,
        sub_dir: Path):
    return BucketFSModelSpecification(model_name=model_specification.model_name,
                                      task_type=model_specification.task_type,
                                      bucketfs_conn_name=bucketfs_conn_name,
                                      sub_dir=sub_dir)
