"""Class ModelSpecification describing a specific model."""

import pathlib
from dataclasses import dataclass
from pathlib import (
    Path,
    PurePosixPath,
)

import transformers


@dataclass(frozen=True)
class ModelTypeData:
    model_factory_dict = {
        "fill-mask": transformers.AutoModelForMaskedLM,
        "translation": transformers.AutoModelForSeq2SeqLM,
        "zero-shot-classification": transformers.AutoModelForSequenceClassification,
        "text-classification": transformers.AutoModelForSequenceClassification,
        "question-answering": transformers.AutoModelForQuestionAnswering,
        "text-generation": transformers.AutoModelForCausalLM,
        "token-classification": transformers.AutoModelForTokenClassification,
    }


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
        switches user input to transformers task types.
        Allows for use of dash or underscore.
        Raises a ValueError if given task_type is not recognized.
        """
        # todo this changes makes loading and installing models of unkown task_type impossible, but allows for listing and deleting.
        #  unkown task_type includes models saved with out old task_types, which we now do not accept anymore.
        #  i though since we have breaking changes anyway, this is the moment to do it. if not i will revert to allowing it always
        # todo check if docu needs updating, add info about ls/delete of legacy task_types?
        allowed_task_types = [
            "fill-mask",
            "translation",
            "zero-shot-classification",
            "text-classification",
            "question-answering",
            "text-generation",
            "token-classification",
        ]
        text_replace_underscore = text.replace("_", "-")
        if text_replace_underscore in allowed_task_types:
            task_type = text_replace_underscore
        else:
            raise ValueError(
                "task_type needs to be one of %s. Refer to the user guide for more information. Found task_type was '%s'"
                % (allowed_task_types, text)
            )
        return task_type

    def legacy_set_task_type_from_udf_name(self, text):
        """
        allows for task_type to be unknown. needed for model listing and deletion of models saved with unknown task_types.
        switches user input(matching udf name) to transformers task types
        """
        if text == "ai_fill_mask_extended":
            task_type = "fill-mask"
        elif text == "question_answering":
            task_type = "question-answering"
        elif text == "sequence_classification":
            task_type = "text-classification"
        elif text == "ai_complete_extended":
            task_type = "text-generation"
        elif text == "ai_extract_extended":
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
        """Returns path suffix specific to the model"""
        return PurePosixPath(
            self.model_name.replace(".", "_") + "_" + self.task_type
        )  # model_name-version-task

    def get_model_factory(self):
        """
        sets model factory depending on the task_type of the specific model
        """
        model_task_type = self.task_type
        try:
            model_factory = ModelTypeData.model_factory_dict[model_task_type]
        except KeyError:
            model_factory = transformers.AutoModel

        return model_factory


def split_path_using_subdir(
    path_parts: tuple[str, ...], model_path: pathlib.Path, sub_dir: str
) -> tuple[str, str]:
    # many models have a name like creator-name/model-name or similar.
    # but we do not know the format exactly.
    # therefor we assume the directory which includes the config.json file to be
    # the model_specific_path_suffix, and everything between this and the sub-dir
    # to be the model_name_prefix
    try:
        subdir_index = path_parts.index(sub_dir)
    except ValueError as e:
        error_message = (
            "subdir not found in path, or subdir is empty string. "
            "given subdir is: %s, not found in path: %s",
            sub_dir,
            model_path,
        )
        raise ValueError(error_message) from e
    name_prefix = "/".join(path_parts[subdir_index + 1 : -1])
    model_specific_path_suffix = path_parts[-1].split(".")[0]
    return name_prefix, model_specific_path_suffix


def best_guess_model_specs(
    model_specific_path_suffix, name_prefix
) -> tuple[str, str, str]:
    # if no known_task_type was found, our best guess is to split the
    # model_specific_path_suffix on "_" and select task_type and model_name accordingly,
    # because we know the model_specific_path_suffix includes at least one "_"
    # followed by the task_type. This might create wrong results if the user
    # choose to use a task_type containing a "_".
    # Note: we dont allow for the saving of models like this anymore, but this stays here in case of legacy models
    warning = (
        "WARNING: We found a model which was saved using a task_type we don't recognize. "
        "As a result, we can only give a best guess on how to parse the model_type and task."
    )
    try:
        model_specific_path_suffix_split = model_specific_path_suffix.split("_")
        if len(model_specific_path_suffix_split) > 1:
            model_name = "/".join(
                [name_prefix, "".join(model_specific_path_suffix_split[0:-1])]
            )
            task_type = model_specific_path_suffix_split[-1]
        return model_name, task_type, warning
    except:
        error_message = (
            "couldn't find a task type in path suffix %s" % model_specific_path_suffix
        )
        raise ValueError(error_message)


def get_task_and_model_name(found_task_types, model_specific_path_suffix, name_prefix):
    task_type = ""
    model_name = ""
    warning = None
    if not found_task_types:
        try:
            model_name, task_type, warning = best_guess_model_specs(
                model_specific_path_suffix, name_prefix
            )
        except ValueError as e:
            raise e

    # if we found known_task_type in the path, check if one is on the end of the
    # model_specific_path_suffix, and declare this one as the task_type.
    # disregard found_task_types form other positions in the model_specific_path_suffix
    for found_task_type in found_task_types:
        if model_specific_path_suffix.endswith("_" + found_task_type):
            model_name = "/".join(
                [
                    name_prefix,
                    model_specific_path_suffix.removesuffix("_" + found_task_type),
                ]
            )
            task_type = found_task_type
            break

    if not task_type or not model_name:
        try:
            model_name, task_type, warning = best_guess_model_specs(
                model_specific_path_suffix, name_prefix
            )
        except ValueError as e:
            raise e
    return model_name, task_type, warning


def create_model_specs_from_path(
    model_path: pathlib.Path, sub_dir
) -> tuple[ModelSpecification, str]:
    path_parts = model_path.parts
    warning = None

    try:
        name_prefix, model_specific_path_suffix = split_path_using_subdir(
            path_parts, model_path, sub_dir
        )
    except ValueError as e:
        raise e

    # find known task_types in the model_specific_path_suffix:
    found_task_types = [
        key
        for key in ModelTypeData.model_factory_dict.keys()
        if key in model_specific_path_suffix
    ]

    try:
        model_name, task_type, warning = get_task_and_model_name(
            found_task_types, model_specific_path_suffix, name_prefix
        )
    except ValueError as e:
        raise e

    if warning:
        # if task_type is not allowed for model_specification, use a placeholder for creation
        # and then replace using the legacy_set_task_type_from_udf_name.
        # needed to allow for deletion of already installed models with illegal task_types
        found_model = ModelSpecification(model_name, "fill-mask")
        found_model.task_type = found_model.legacy_set_task_type_from_udf_name(
            task_type
        )
        return found_model, warning

    return ModelSpecification(model_name, task_type), warning
