from inspect import cleandoc

import exasol.bucketfs as bfs
import pyexasol
import pytest
import transformers as huggingface

from exasol_transformers_extension.utils.model_specification import ModelSpecification
from exasol_transformers_extension.utils.model_utils import install_huggingface_model

# original implementation used
# from exasol.ai.text.impl.nlp.algorithms.udf_hft_base import DEVICE_CPU

@pytest.fixture(scope="session")
def xbucketfs_location() -> bfs.path.PathLike:
    import os
    password = os.getenv("BUCKETFS_PASSWORD")
    return bfs.path.build_path(
            backend="onprem",
            url="http://192.168.124.221:2580",
            username="w",
            password=password,
            service_name="bfsdefault",
            bucket_name="default",
            verify=False,
            path="",
    )

@pytest.fixture(scope="session")
def xdb_conn():
    return pyexasol.connect(dsn="192.168.124.221:8563", user="sys", password="exasol")

DEVICE_CPU = -1


@pytest.fixture(scope="session")
def xsetup_database(pyexasol_connection):
    return "", ""


def xtest_x1(db_conn, bucketfs_location):
    result = db_conn.execute(
        "SELECT {value}",
        query_params={"value": "Hello"},
    ).fetchone()
    for f in bucketfs_location.iterdir():
        print(f'{f}')
    bfspath = bucketfs_location / "sub_dir" / "a.html"
    with open("/home/chku/tmp/a.html", "br") as file:
        bfspath.write(file)
    print(f'uploaded to {bfspath}')


def test_install_huggingface_model(
    bucketfs_location: bfs.path.PathLike,
    setup_database,
    db_conn: pyexasol.ExaConnection
):
    bucketfs_conn_name, _ = setup_database
    mspec = ModelSpecification("t5-small", "translation")
    sub_dir = "sub_dir"
    install_huggingface_model(
        bucketfs_location=bucketfs_location,
        sub_dir=sub_dir,
        task_type=mspec.task_type,
        model_name=mspec.model_name,
        model_factory=mspec.get_model_factory(),
        tokenizer_factory=huggingface.AutoTokenizer,
        huggingface_token=None,
    )
    # TODO: create UDF
    # TODO: create CONNECTION
    query = cleandoc(
        """
        SELECT TE_MODEL_LOADER_UDF(
          '{mspec.model_name}',
          '{mspec.task_type}',
          '{sub_dir}',
          '{bucketfs_conn_name}'
        )
        """
    )
    result = db_conn.execute(query).fetchone()
    print(f'{result}')
    # assert result == [mspec.task_type, ]
