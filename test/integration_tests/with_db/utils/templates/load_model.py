# This file is to be included into UDF definition in the adjacent file
# load_model.jinja.sql used in integration tests.

from pathlib import Path

from exasol_transformers_extension.utils.bucketfs_model_specification import (
    BucketFSModelSpecification,
)
from exasol_transformers_extension.utils.model_utils import load_huggingface_pipeline

DEVICE_CPU = -1


def run(ctx):
    mspec = BucketFSModelSpecification(
        model_name=ctx.model_name,
        task_type=ctx.task_type,
        bucketfs_conn_name=ctx.bfs_conn,
        sub_dir=Path(ctx.sub_dir),
    )
    load_huggingface_pipeline(exa, model_spec=mspec, device=DEVICE_CPU)
    return True
