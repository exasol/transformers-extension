from dataclasses import dataclass
from pathlib import Path

from exasol_transformers_extension.utils.current_model_specification import CurrentModelSpecification
from exasol_transformers_extension.utils.model_specification_string import ModelSpecificationString


@dataclass(frozen=True)
class BucketFSParams:
    real_port: str
    name: str
    bucket: str
    path_in_bucket: str


@dataclass(frozen=True)
class ModelParams:
    base_model_specs: ModelSpecificationString
    seq2seq_model_specs: ModelSpecificationString
    tiny_model_specs: ModelSpecificationString
    text_data: str
    sub_dir: str


bucketfs_params = BucketFSParams(
    real_port="6583",
    name="bfsdefault",
    bucket="default",
    path_in_bucket="container")

model_params = ModelParams(
    base_model_specs=ModelSpecificationString('bert-base-uncased'),
    seq2seq_model_specs=ModelSpecificationString("t5-small"),
    tiny_model_specs=ModelSpecificationString("prajjwal1/bert-tiny"),
    text_data='The company Exasol is based in Nuremberg',
    sub_dir='model_sub_dir')
