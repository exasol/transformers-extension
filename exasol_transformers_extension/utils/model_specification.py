from pathlib import PurePosixPath, Path

class ModelSpecification:
    """
    Class describing a model.
    """
    def __init__(self, model_name: str):
        # task_type, model_version
        self.model_name = model_name

    def get_model_specs_for_download(self):
        # returns all attributes necessary for downloading the model from Huggingface
        return self.model_name

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, ModelSpecification):
            return self.model_name == other.model_name
        return False

    def get_model_specific_path_suffix(self) -> PurePosixPath:
        return PurePosixPath(self.model_name) #model_name-version-task

