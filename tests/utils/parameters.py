from dataclasses import dataclass
from pathlib import Path

from exasol_transformers_extension.utils.current_model_specification import CurrentModelSpecification
from exasol_transformers_extension.utils.model_specification import ModelSpecification


@dataclass(frozen=True)
class BucketFSParams:
    real_port: str
    name: str
    bucket: str
    path_in_bucket: str


@dataclass(frozen=True)
class ModelParams:
    base_model_specs: ModelSpecification #this is used for other tests, task_type should be set per test
    seq2seq_model_specs: ModelSpecification #tis model is used for testing translation_udf
    tiny_model_specs: ModelSpecification #this model is used for upload/download tests
    text_data: str
    sub_dir: str


bucketfs_params = BucketFSParams(
    real_port="6583",
    name="bfsdefault",
    bucket="default",
    path_in_bucket="container")

model_params = ModelParams(
    base_model_specs=ModelSpecification('bert-base-uncased', "need to set this task_type"),
    seq2seq_model_specs=ModelSpecification("t5-small", "translation"),
    tiny_model_specs=ModelSpecification("prajjwal1/bert-tiny", ""),#todo make work with empty task_zype or use a real one?
    text_data='The company Exasol is based in Nuremberg',
    sub_dir='model_sub_dir')
