from pathlib import Path

from exasol_transformers_extension.utils.bucketfs_model_specification import (
    BucketFSModelSpecificationFactory,
)

model_spec_factory = BucketFSModelSpecificationFactory()

DEFAULT_SUBDIR = "TE_default_models"
DEFAULT_BUCKETFS_CONN_NAME = "EXA_AI_FUNCTION_MODEL_LOCATION"

DEFAULT_MODEL_SPECS = {
    # these are placeholder model specs, remove them once we have decided on a real one
    "model_for_a_specific_udf": model_spec_factory.create(
        model_name="prajjwal1/bert-tiny",
        task_type="fill_mask",
        bucketfs_conn_name=DEFAULT_BUCKETFS_CONN_NAME,
        sub_dir=Path(DEFAULT_SUBDIR),
    ),
    "model_for_another_udf": model_spec_factory.create(
        model_name="prajjwal1/bert-tiny",
        task_type="fill_mask",
        bucketfs_conn_name=DEFAULT_BUCKETFS_CONN_NAME,
        sub_dir=Path(DEFAULT_SUBDIR),
    ),
}
