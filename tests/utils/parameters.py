from dataclasses import dataclass


@dataclass(frozen=True)
class DBParams:
    host: str
    port: str
    user: str
    password: str

    def address(self) -> str:
        return f"{self.host}:{self.port}"


@dataclass(frozen=True)
class BucketFSParams:
    host: str
    port: str
    real_port: str
    user: str
    password: str
    name: str
    bucket: str
    path_in_bucket: str

    def address(self, port=None) -> str:
        port = self.port if not port else port
        return f"http://{self.host}:{port}/{self.bucket}/" \
               f"{self.path_in_bucket};{self.name}"


@dataclass(frozen=True)
class ModelParams:
    base_model: str
    seq2seq_model: str
    tiny_model: str
    text_data: str
    sub_dir: str


db_params = DBParams(
    host="127.0.0.1",
    port="9563",
    user="sys",
    password="exasol")

bucketfs_params = BucketFSParams(
    host="127.0.0.1",
    port="6666",
    real_port="2580",
    user="w",
    password="write",
    name="bfsdefault",
    bucket="default",
    path_in_bucket="container")

model_params = ModelParams(
    base_model='bert-base-uncased',
    seq2seq_model="t5-small",
    tiny_model="prajjwal1/bert-tiny",
    text_data='The company Exasol is based in Nuremberg',
    sub_dir='model_sub_dir')
