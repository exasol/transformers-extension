from dataclasses import dataclass


@dataclass(frozen=True)
class BucketFSParams:
    real_port: str
    name: str
    bucket: str
    path_in_bucket: str


@dataclass(frozen=True)
class ModelParams:
    base_model: str
    seq2seq_model: str
    tiny_model: str
    text_data: str
    sub_dir: str


bucketfs_params = BucketFSParams(
    real_port="6583",
    name="bfsdefault",
    bucket="default",
    path_in_bucket="container")

model_params = ModelParams( #todo probs just put a ModelSpecificationString in here
    base_model='bert-base-uncased',
    seq2seq_model="t5-small",
    tiny_model="prajjwal1/bert-tiny",
    text_data='The company Exasol is based in Nuremberg',
    sub_dir='model_sub_dir')
