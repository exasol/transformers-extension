import contextlib
from inspect import cleandoc
from pathlib import Path

import exasol.bucketfs as bfs
import jinja2
import pyexasol
import pytest
import transformers as huggingface
from jinja2 import (
    PackageLoader,
    select_autoescape,
)

from exasol_transformers_extension.utils.bucketfs_model_specification import (
    BucketFSModelSpecification,
)
from exasol_transformers_extension.utils.model_utils import install_huggingface_model

# original implementation used
# from exasol.ai.text.impl.nlp.algorithms.udf_hft_base import DEVICE_CPU

DEVICE_CPU = -1


@contextlib.contextmanager
def not_raises(exception):
    try:
        yield
    except exception:
        raise pytest.fail(f"Did raise {exception}")


@pytest.fixture
def load_model_udf(
    language_alias: str,
    db_schema_name: str,
    db_conn: pyexasol.ExaConnection,
):
    jenv = jinja2.Environment(
        loader=PackageLoader(__name__),
        autoescape=select_autoescape(),
    )
    python = jenv.get_template("load_model.py").render()
    udf = jenv.get_template("load_model.jinja.sql").render(
        language_alias=language_alias,
        schema=db_schema_name,
        script_content=python,
    )
    db_conn.execute(udf)


def test_install_and_load_huggingface_model(
    load_model_udf,
    bucketfs_location: bfs.path.PathLike,
    setup_database,
    db_conn: pyexasol.ExaConnection,
    db_schema_name: str,
):
    """
    This test actually uses load_huggingface_pipeline() to verify the
    validity of the installed model.
    """
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
        SELECT "{db_schema_name}"."TE_LOAD_MODEL"(
          '{mspec.model_name}',
          '{mspec.task_type}',
          '{mspec.sub_dir}',
          '{bucketfs_conn_name}'
        )
        """
    )
    with not_raises(Exception):
        db_conn.execute(query)
