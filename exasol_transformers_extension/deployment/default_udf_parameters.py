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
        model_name="tabularisai/robust-sentiment-analysis",
        task_type="text-classification",
        bucketfs_conn_name=DEFAULT_BUCKETFS_CONN_NAME,
        sub_dir=Path(DEFAULT_SUBDIR),
    ),
    "AiClassifyUDF": model_spec_factory.create(
        model_name="MoritzLaurer/ModernBERT-large-zeroshot-v2.0",
        task_type="zero-shot-classification",
        bucketfs_conn_name=DEFAULT_BUCKETFS_CONN_NAME,
        sub_dir=Path(DEFAULT_SUBDIR),
    ),
    "AiExtractEntitiesUDF": model_spec_factory.create(
        model_name="guishe/nuner-v2_fewnerd_fine_super",
        task_type="token-classification",
        bucketfs_conn_name=DEFAULT_BUCKETFS_CONN_NAME,
        sub_dir=Path(DEFAULT_SUBDIR),
    ),
}
