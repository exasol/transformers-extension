from dataclasses import dataclass
from pathlib import Path

from exasol_transformers_extension.utils.bucketfs_model_specification import BucketFSModelSpecification
from exasol_transformers_extension.utils.model_specification import ModelSpecification


@dataclass(frozen=True)
class BucketFSParams:
    real_port: str
    name: str
    bucket: str
    path_in_bucket: str


@dataclass(frozen=True)
class ModelParams:
    base_model_specs: ModelSpecification        # this is used for other tests, taks_name should be set per test
    seq2seq_model_specs: ModelSpecification     # this model is used for testing translation_udf
    q_a_model_specs: ModelSpecification         # this model is used for testing question answering
    text_gen_model_specs: ModelSpecification    # used for text generation tests
    token_model_specs: ModelSpecification       # this model is used for token classification tests
    sequence_class_model_specs: ModelSpecification          # this model is used for sequence classification single text tests
    sequence_class_pair_model_specs: ModelSpecification     # this model is used for sequence classification text pair tests
    zero_shot_model_specs: ModelSpecification   # this model is used for zero-shot-classification tests
    tiny_model_specs: ModelSpecification        # this model is used for upload/download tests
    text_data: str
    sub_dir: str


bucketfs_params = BucketFSParams(
    real_port="6583",
    name="bfsdefault",
    bucket="default",
    path_in_bucket="container")

model_params = ModelParams(
    base_model_specs=ModelSpecification('bert-base-uncased', "need to set this task_type"), #fill mask
    seq2seq_model_specs=ModelSpecification("t5-small", "translation"),
    q_a_model_specs=ModelSpecification("deepset/tinybert-6l-768d-squad2", "question-answering"),
    text_gen_model_specs=ModelSpecification("openai-community/gpt2", "text-generation"),
    token_model_specs=ModelSpecification("dslim/bert-base-NER", "token-classification"),#token-classification
    sequence_class_model_specs=ModelSpecification("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis", "text-classification"),
    sequence_class_pair_model_specs=ModelSpecification("MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli", "text-classification"),#Alireza1044/albert-base-v2-mnli
    zero_shot_model_specs=ModelSpecification("MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33", "zero-shot-classification"),#text-class
    tiny_model_specs=ModelSpecification("prajjwal1/bert-tiny", "task"),
    text_data='The database software company Exasol is based in Nuremberg',
    sub_dir='model_sub_dir')
