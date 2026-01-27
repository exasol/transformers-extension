from __future__ import annotations

from dataclasses import dataclass

from exasol_transformers_extension.utils.model_specification import ModelSpecification

PATH_IN_BUCKET = "container"


@dataclass(frozen=True)
class ModelParams:
    base_model_specs: ModelSpecification  # this is used for other test, taks_name should be set per test
    seq2seq_model_specs: (
        ModelSpecification  # this model is used for testing translation_udf
    )
    q_a_model_specs: (
        ModelSpecification  # this model is used for testing question answering
    )
    text_gen_model_specs: ModelSpecification  # used for text generation test
    token_model_specs: (
        ModelSpecification  # this model is used for token classification test
    )
    text_classification_model_specs: ModelSpecification  # this model is used for text classification single text test
    text_classification_pair_model_specs: (
        ModelSpecification  # this model is used for ai_entailment_extended test
    )
    zero_shot_model_specs: (
        ModelSpecification  # this model is used for zero-shot-classification test
    )
    tiny_model_specs: ModelSpecification  # this model is used for upload/download test
    text_data: str
    sub_dir: str
    ls_test_subdir: str


model_params = ModelParams(
    base_model_specs=ModelSpecification(
        "bert-base-uncased", "need to set this task_type"
    ),
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
    tiny_model_specs=ModelSpecification("prajjwal1/bert-tiny", "task"),
    text_data="The database software company Exasol is based in Nuremberg",
    sub_dir="model_sub_dir",
    ls_test_subdir="ls_test_subdir",
)
