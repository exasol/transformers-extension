"""
Caller for AiTranslateExtendedUDF
"""

from exasol_transformers_extension.udfs.models.ai_translate_extended_udf import (
    AiTranslateExtendedUDF,
)

udf = AiTranslateExtendedUDF(exa)


def run(ctx):
    """
    run function for AiTranslateExtendedUDF
    """
    return udf.run(ctx)
