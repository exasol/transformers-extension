from __future__ import annotations
from dataclasses import dataclass

from exasol_transformers_extension.utils.model_specification import ModelSpecification


@dataclass(frozen=True)
class BucketFSParams:
    real_port: str
    name: str
    bucket: str
    path_in_bucket: str


@dataclass(frozen=True)
class ModelParams:
    base_model_specs: ModelSpecification
    seq2seq_model_specs: ModelSpecification
    tiny_model_specs: ModelSpecification
    text_data: str
    sub_dir: str


bucketfs_params = BucketFSParams(
    real_port="6583",
    name="bfsdefault",
    bucket="default",
    path_in_bucket="container")

model_params = ModelParams(
    base_model_specs=ModelSpecification('bert-base-uncased'),
    seq2seq_model_specs=ModelSpecification("t5-small"),
    tiny_model_specs=ModelSpecification("prajjwal1/bert-tiny"),
    text_data='The company Exasol is based in Nuremberg',
    sub_dir='model_sub_dir')


def get_arg_list(**kwargs) -> list[str]:
    args_list: list[str] = []
    for k, v in kwargs.items():
        args_list.append(f'--{k.replace("_", "-")}')
        args_list.append(str(v))
    return args_list
