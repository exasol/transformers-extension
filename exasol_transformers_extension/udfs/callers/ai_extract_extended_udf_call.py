"""
Caller for AiExtractExtendedUDF
"""

from exasol_transformers_extension.udfs.models.ai_extract_extended_udf import (
    AiExtractExtendedUDF,
)

udf = AiExtractExtendedUDF(exa)


def run(ctx):
    """
    run function for AiExtractExtendedUDF
    """
    return udf.run(ctx)
