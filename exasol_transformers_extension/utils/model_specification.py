from pathlib import PurePosixPath, Path
import transformers


class ModelSpecification:
    """
    Class describing a model.
    """
    def __init__(self, model_name: str, task_type: str):
        # task_type, model_version
        self.model_name = model_name
        self.task_type = self._set_task_type_from_udf_name(task_type)

    def _set_task_type_from_udf_name(self, text):
        """
        switches user input(matching udf name) to transformers task types
        """
        if text == "filling_mask":
            task_type = "fill-mask"
        elif text == "question_answering":
            task_type = "question-answering"
        elif text == "sequence_classification":
            task_type = "text-classification"
        elif text == "text_generation":
            task_type = "text-generation"
        elif text == "token_classification":
            task_type = "token-classification"
        elif text == "translation":
            task_type = "translation"
        elif text == "zero_shot_classification":
            task_type = "zero-shot-classification"
        else:
            task_type = text
        return task_type

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, ModelSpecification):
            return (self.model_name == other.model_name
                    and self.task_type == other.task_type)
        return False

    def get_model_specific_path_suffix(self) -> PurePosixPath:
        return PurePosixPath(self.model_name.replace(".", "_") + "_" + self.task_type) #model_name-version-task#

    def get_model_factory(self):
        """
        sets model factory depending on the task_type of the specific model
        """
        model_task_type = self.task_type
        if model_task_type == "fill-mask":
            model_factory = transformers.AutoModelForMaskedLM
        elif model_task_type == "translation":
            model_factory = transformers.AutoModelForSeq2SeqLM
        elif model_task_type == "zero-shot-classification":
            model_factory = transformers.AutoModelForSequenceClassification
        elif model_task_type == "text-classification":
            model_factory = transformers.AutoModelForSequenceClassification
        elif model_task_type == "question-answering":
            model_factory = transformers.AutoModelForQuestionAnswering
        elif model_task_type == "text-generation":
            model_factory = transformers.AutoModelForCausalLM
        elif model_task_type == "token-classification":
            model_factory = transformers.AutoModelForTokenClassification
        else:
            model_factory = transformers.AutoModel
        return model_factory
