from exasol_transformers_extension.utils.model_specification import ModelSpecification
from pathlib import PurePosixPath, Path

class CurrentModelSpecification(ModelSpecification):
    """
    Class describing a model with additional information about
    the bucketFS connection and the subdir in the bucketfs the model can be found at.
    """
    def __init__(self,
                 model_name: str,
                 bucketfs_conn_name: str,
                 sub_dir: Path):
        ModelSpecification.__init__(self, model_name)
        self.bucketfs_conn_name = bucketfs_conn_name
        self.sub_dir = sub_dir

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, CurrentModelSpecification):
            return (super().__eq__(other)
                    and self.sub_dir == other.sub_dir and
                    self.bucketfs_conn_name == other.bucketfs_conn_name)
        return False

    def get_bucketfs_model_save_path(self) -> Path:
        """
        path model is saved at in the bucketfs
        """
        model_name = self.get_model_specific_path_suffix()
        return Path(self.sub_dir, model_name)



class CurrentModelSpecificationFactory:
    def create(self,
               model_name: str,
               bucketfs_conn_name: str,
               sub_dir: Path):
        return CurrentModelSpecification(model_name, bucketfs_conn_name, sub_dir)


class CurrentModelSpecificationFromModelSpecs:
    def transform(self,
                  model_specification: ModelSpecification,
                  bucketfs_conn_name: str,
                  sub_dir: Path):
        return CurrentModelSpecification(model_name=model_specification.model_name,
                                         bucketfs_conn_name=bucketfs_conn_name,
                                         sub_dir=sub_dir)
