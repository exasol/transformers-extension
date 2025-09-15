from inspect import cleandoc
from pathlib import Path

import exasol.bucketfs as bfs
import pyexasol
import transformers as huggingface

from exasol_transformers_extension.utils.bucketfs_model_specification import (
    BucketFSModelSpecification,
)
from exasol_transformers_extension.utils.model_utils import install_huggingface_model

# original implementation used
# from exasol.ai.text.impl.nlp.algorithms.udf_hft_base import DEVICE_CPU

DEVICE_CPU = -1

'''
import pytest

@pytest.fixture(scope="session")
def bucketfs_location() -> bfs.path.PathLike:
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
def db_conn():
    return pyexasol.connect(dsn="192.168.124.221:8563", user="sys", password="exasol")


@pytest.fixture(scope="session")
def setup_database(db_conn):
    return "TEST_TE_BFS_CONNECTION", ""


def test_x1(db_conn, bucketfs_location):
    result = db_conn.execute(
        "SELECT {value}",
        query_params={"value": "Hello"},
    ).fetchone()
    for f in bucketfs_location.iterdir():
        print(f"{f}")
    bfspath = bucketfs_location / "sub_dir" / "a.html"
    with open("/home/chku/tmp/a.html", "br") as file:
        bfspath.write(file)
    print(f"uploaded to {bfspath}")


@pytest.fixture(scope="session")
def db_schema_name() -> str:
    return "TE_ITEST"


def test_x2(db_schema_name, db_conn):
    print(f"{db_schema_name}")
    query = cleandoc(
        """
        SELECT {db_schema_name}TE_MODEL_LOADER_UDF(
          '{mspec.model_name}',
          '{mspec.task_type}',
          '{sub_dir}',
          '{bucketfs_conn_name}'
        )
        """
    )
    result = db_conn.execute(query).fetchone()
    print(f"{result}")
'''


def test_install_huggingface_model(
    bucketfs_location: bfs.path.PathLike,
    setup_database,
    db_conn: pyexasol.ExaConnection,
    db_schema_name: str,
):
    bucketfs_conn_name, _ = setup_database
    mspec = BucketFSModelSpecification(
        model_name="t5-small",
        task_type="translation",
        bucketfs_conn_name=bucketfs_conn_name,
        sub_dir=Path("sub_dir"),
    )
    install_huggingface_model(
        bucketfs_location=bucketfs_location,
        model_spec=mspec,
        tokenizer_factory=huggingface.AutoTokenizer,
        huggingface_token=None,
    )
    query = cleandoc(
        f"""
        SELECT "{db_schema_name}"."TE_MODEL_LOADER_UDF"(
          '{mspec.model_name}',
          '{mspec.task_type}',
          '{mspec.sub_dir}',
          '{bucketfs_conn_name}'
        )
        """
    )
    result = db_conn.execute(query).fetchone()
    print(f"{result}")
    # assert result == [mspec.task_type, ]
