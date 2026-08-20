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
    "max_new_tokens": 256,  # todo document!
    "return_full_text": False,
    "aggregation_strategy": "simple",
}

DEFAULT_MODEL_SPECS = {
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
    "AiTranslateUDF": model_spec_factory.create(
        model_name="google-t5/t5-small",  # todo?
        task_type="translation",
        bucketfs_conn_name=DEFAULT_BUCKETFS_CONN_NAME,
        sub_dir=Path(DEFAULT_SUBDIR),
    ),
    "AiAnswerUDF": model_spec_factory.create(
        model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
        task_type="text-generation",
        bucketfs_conn_name=DEFAULT_BUCKETFS_CONN_NAME,
        sub_dir=Path(DEFAULT_SUBDIR),
    ),
}
