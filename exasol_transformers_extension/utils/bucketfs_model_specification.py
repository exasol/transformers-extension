from pathlib import (
    Path,
    PurePosixPath,
)

from exasol_transformers_extension.utils.model_specification import ModelSpecification


class BucketFSModelSpecification(ModelSpecification):
    """
    Describes a model with additional information about the BucketFS
    connection and the subdir in the BucketFS the model should be uploaded to
    or can be found at after uploading.
    """

    def __init__(
        self, model_name: str, task_type: str, bucketfs_conn_name: str, sub_dir: Path
    ):
        """
        model_name:
            Name of the model. This is the same name as it's seen on the Haggingface
            model card, for example 'cross-encoder/nli-deberta-base'.
        task_type:
            Name of an NLP task recognized by the huggingface.pipeline(). See
            https://huggingface.co/docs/transformers/v4.48.2/en/main_classes/pipelines#transformers.pipeline.task
        bucketfs_conn_name:
            Name of the BucketFS connection to retrieve the BucketFS location from.
        sub_dir:
            Subdirectory in the BucketFS where the model can be found at.
        """
        ModelSpecification.__init__(self, model_name, task_type)
        self.bucketfs_conn_name = bucketfs_conn_name
        self.sub_dir = sub_dir

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, BucketFSModelSpecification):
            return (
                super().__eq__(other)
                and self.sub_dir == other.sub_dir
                and self.bucketfs_conn_name == other.bucketfs_conn_name
            )
        return False

    def get_bucketfs_model_save_path(self) -> Path:
        """
        path model is saved at in the bucketfs
        """
        model_path_suffix = self.get_model_specific_path_suffix()
        return Path(self.sub_dir, model_path_suffix)


class BucketFSModelSpecificationFactory:
    def create(
        self, model_name: str, task_type: str, bucketfs_conn_name: str, sub_dir: Path
    ):
        return BucketFSModelSpecification(
            model_name, task_type, bucketfs_conn_name, sub_dir
        )


def get_BucketFSModelSpecification_from_model_Specs(
    model_specification: ModelSpecification, bucketfs_conn_name: str, sub_dir: Path
):
    return BucketFSModelSpecification(
        model_name=model_specification.model_name,
        task_type=model_specification.task_type,
        bucketfs_conn_name=bucketfs_conn_name,
        sub_dir=sub_dir,
    )
