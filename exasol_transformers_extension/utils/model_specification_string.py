from pathlib import PurePosixPath, Path

class ModelSpecificationString:
    """
    Class describing a model.
    """
    def __init__(self, model_name: str):
        # task_type, model_version
        self.model_name = model_name

    def deconstruct(self):
        return self.model_name

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, ModelSpecificationString):
            return self.model_name == other.model_name
        return False

    def get_model_specific_path_suffix(self) -> PurePosixPath: #todo use
        return PurePosixPath(self.model_name) #model_name-version-task

