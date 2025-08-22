"""Class ModelSpecification describing a specific model."""

from pathlib import PurePosixPath, Path

import transformers


class ModelSpecification:
    """
    Class describing a model.
    """

    def __init__(self, model_name: str, task_type: str):
        """
        :param model_name: Name of the model
        :param task_type: Name of the task model is intended for
        """
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
        """Overrides the default implementation of equal"""
        if isinstance(other, ModelSpecification):
            return (
                self.model_name == other.model_name
                and self.task_type == other.task_type
            )
        return False

    def get_model_specific_path_suffix(self) -> PurePosixPath:
        """Returns pyth suffix specific to the model"""
        return PurePosixPath(
            self.model_name.replace(".", "_") + "_" + self.task_type
        )  # model_name-version-task#todo just replace the slashes too?


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

def create_model_spcs_from_path(model_path: Path, sub_dir) -> ModelSpecification:
    #todo or do we just make it so you ned subdir to save? or collect model specs in a file somewhere? and the just check if still correct?
    # this could  get out of sinc if user uses something different to delete or upload nmodels
        path_parts = model_path.parts
        try:
            subdir_index = path_parts.index(sub_dir)#todo what do we return if no subdir is given? or add "find all subdirs" function?
            #todo return error if subdir not in path_parts, what happens if models not in subdir, or subdir = ""

        # many models have a name like creator-name/model-name or similar. but we do not know the format exactly.
        # therefor we assume the name of the tar file to be the model_specific_path_suffix,
        # and everything between this and the sub-dir to be the model_name_prefix
        except:
            subdir_index = -2#todo if there is only modelname this will create wrong results
        name_prefix = "/".join(path_parts[subdir_index+1:-1])
        model_specific_path_suffix = path_parts[-1].split('.')[0]
        # we know the model_specific_path_suffix includes at least on "_" followed by the task_name
        model_specific_path_suffix_split = model_specific_path_suffix.split("_")
        model_name = "/".join([name_prefix, "".join(model_specific_path_suffix_split[0:-1])])# todo cant easily convert "_" back to ".", cause dont know which ones
        #todo do we re-switch the task type? -> create model spec parameters task_name and "inernal_task_name/transformers_task_name"
        task_name = model_specific_path_suffix_split[-1]
        return ModelSpecification(model_name, task_name)
