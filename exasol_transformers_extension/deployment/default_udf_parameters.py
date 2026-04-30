from pathlib import Path

from exasol_transformers_extension.utils.bucketfs_model_specification import (
    BucketFSModelSpecificationFactory,
)

model_spec_factory = BucketFSModelSpecificationFactory()

DEFAULT_SUBDIR = "TE_default_models"
DEFAULT_BUCKETFS_CONN_NAME = "EXA_AI_MODEL_LOCATION"

DEFAULT_VALUES = {
    "sub_dir": DEFAULT_SUBDIR,
    "bucketfs_conn": DEFAULT_BUCKETFS_CONN_NAME,
    "device_id": None,
    "top_k": 1,
    "return_ranks": "HIGHEST",
    "max_new_tokens": None,
    "return_full_text": False,
    "aggregation_strategy": "simple",
}

DEFAULT_MODEL_SPECS = {
    # these are placeholder model specs, remove them once we have decided on a real one
    "AiSentimentUDF": model_spec_factory.create(
        model_name="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
        task_type="text-classification",
        bucketfs_conn_name=DEFAULT_BUCKETFS_CONN_NAME,
        sub_dir=Path(DEFAULT_SUBDIR),
    ),
    "model_for_another_udf": model_spec_factory.create(
        model_name="prajjwal1/bert-tiny",
        task_type="different_task",
        bucketfs_conn_name=DEFAULT_BUCKETFS_CONN_NAME,
        sub_dir=Path(DEFAULT_SUBDIR),
    ),
}
