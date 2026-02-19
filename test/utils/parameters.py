from __future__ import annotations

from dataclasses import dataclass
from textwrap import fill

from exasol_transformers_extension.utils.model_specification import ModelSpecification

PATH_IN_BUCKET = "container"


@dataclass(frozen=True)
class ModelParams:
    fill_model_specs: ModelSpecification  # this is used for fill_mask model tests
    seq2seq_model_specs: (
        ModelSpecification  # this model is used for testing ai_translate_extended udf
    )
    q_a_model_specs: (
        ModelSpecification  # this model is used for testing ai_answer_extended udf
    )
    text_gen_model_specs: ModelSpecification  # used for ai_complete_extended test
    token_model_specs: (
        ModelSpecification  # this model is used for ai_extract_extended test
    )
    text_classification_model_specs: ModelSpecification  # this model is used for text classification single text test
    text_classification_pair_model_specs: (
        ModelSpecification  # this model is used for ai_entailment_extended test
    )
    zero_shot_model_specs: (
        ModelSpecification  # this model is used for ai_classify_extended test
    )
    tiny_model_specs: ModelSpecification  # this model is used for upload/download test
    # a model with a task_type not recognized by us. used for testing ls and delete of legacy models
    illegal_tiny_model_specs: ModelSpecification
    text_data: str
    sub_dir: str
    ls_test_subdir: str


def create_illegal_tiny_model_specs():
    illegal_tiny_model_specs = ModelSpecification("prajjwal1/bert-tiny", "fill_mask")
    illegal_tiny_model_specs.task_type = (
        illegal_tiny_model_specs.legacy_set_task_type_from_udf_name("illegal-task-type")
    )
    return illegal_tiny_model_specs


model_params = ModelParams(
    fill_model_specs=ModelSpecification("bert-base-uncased", "fill_mask"),
    seq2seq_model_specs=ModelSpecification("t5-small", "translation"),
    q_a_model_specs=ModelSpecification(
        "deepset/tinybert-6l-768d-squad2", "question-answering"
    ),
    text_gen_model_specs=ModelSpecification("openai-community/gpt2", "text-generation"),
    token_model_specs=ModelSpecification("dslim/bert-base-NER", "token-classification"),
    text_classification_model_specs=ModelSpecification(
        "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
        "text-classification",
    ),
    text_classification_pair_model_specs=ModelSpecification(
        "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli", "text-classification"
    ),
    zero_shot_model_specs=ModelSpecification(
        "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33",
        "zero-shot-classification",
    ),
    tiny_model_specs=ModelSpecification("prajjwal1/bert-tiny", "fill-mask"),
    illegal_tiny_model_specs=create_illegal_tiny_model_specs(),
    text_data="The database software company Exasol is based in Nuremberg",
    sub_dir="model_sub_dir",
    ls_test_subdir="ls_test_subdir",
)
