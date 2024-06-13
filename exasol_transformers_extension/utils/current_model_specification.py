from exasol_transformers_extension.utils.model_specification_string import ModelSpecificationString
from pathlib import PurePosixPath, Path

class CurrentModelSpecification(ModelSpecificationString):
    """
    Class describing a model with additional information about
    the bucketFS connection and the subdir in the bucketfs the model can be found at.
    """
    def __init__(self,
                 model_name: str,
                 bucketfs_conn_name: str,
                 sub_dir: Path):
        ModelSpecificationString.__init__(self, model_name)
        #self.model_specification_string = model_specification_string
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
        model_name = self.get_model_specific_path_suffix() #todo change in other path creations
        return Path(self.sub_dir, model_name)
    # todo add class replacing current_model_key includes a ModelSpecification and move path creation functions there


class CurrentModelSpecificationFactory:
    def create(self,
               model_name: str,
               bucketfs_conn_name: str,
               sub_dir: Path):
        return CurrentModelSpecification(model_name, bucketfs_conn_name, sub_dir)


class CurrentModelSpecificationFromModelSpecs:
    def transform(self,
                  model_specification_string: ModelSpecificationString,
                  bucketfs_conn_name: str,
                  sub_dir: Path):
        return CurrentModelSpecification(model_name=model_specification_string.model_name,
                                         bucketfs_conn_name=bucketfs_conn_name,
                                         sub_dir=sub_dir)
