from pathlib import PurePosixPath, Path
import transformers


class ModelSpecification:
    """
    Class describing a model.
    """
    def __init__(self, model_name: str, task_type: str):
        # task_type, model_version
        self.model_name = model_name
        self.task_type = task_type

    def get_model_specs_for_download(self):#todo change usages?
        """
        returns all attributes necessary for downloading the model from Huggingface.
        """
        return self.model_name, self.task_type

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, ModelSpecification):
            return (self.model_name == other.model_name
                    and self.task_type == other.task_type)
        return False

    def get_model_specific_path_suffix(self) -> PurePosixPath:
        return PurePosixPath(self.model_name + "_" + self.task_type) #model_name-version-task

    def get_model_factory(self):
        """
        sets model factory depending on the task_type of the specific model
        """
        model_task_type = self.task_type
        if model_task_type == "filling_mask":
            model_factory = transformers.AutoModelForMaskedLM #todo make switchcase?
        elif model_task_type == "translation":
            model_factory = transformers.T5Model #todo correct?
        else:
            model_factory = transformers.AutoModel
        return model_factory

