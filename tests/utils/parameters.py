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


@dataclass(frozen=True)
class FillingMaskModelParams:
    model1: CurrentModelSpecification
    model2: CurrentModelSpecification
    text_mask1: str
    text_mask2: str


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

filling_mask_model_params = FillingMaskModelParams(
    model1=CurrentModelSpecification("model1",
                                     "bfs_conn1",
                                     Path("sub_dir1")),
    model2=CurrentModelSpecification("model2",
                                     "bfs_conn2",
                                     Path("sub_dir2")),
    text_mask1="text <mask> 1",
    text_mask2="text <mask> 2",
)
