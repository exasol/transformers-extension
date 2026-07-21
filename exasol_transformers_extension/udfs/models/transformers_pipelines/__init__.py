from transformers import AutoModelForSeq2SeqLM
from transformers.pipelines import PIPELINE_REGISTRY

from exasol_transformers_extension.udfs.models.transformers_pipelines.translation import TranslationPipeline

PIPELINE_REGISTRY.register_pipeline(
    task="translation",
    pipeline_class=TranslationPipeline,
    pt_model=AutoModelForSeq2SeqLM,
    type="text",
)