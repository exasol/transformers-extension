"""Caller for AiExtractEntitiesUDF"""

from exasol_transformers_extension.udfs.models.ai_extract_entities_udf import (
    AiExtractEntitiesUDF,
)

udf = AiExtractEntitiesUDF(exa)


def run(ctx):
    """run function for AiExtractEntitiesUDF"""
    return udf.run(ctx)
