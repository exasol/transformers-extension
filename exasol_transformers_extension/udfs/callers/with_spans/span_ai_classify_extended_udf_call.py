"""
Caller for AiClassifyExtendedUDF
"""

from exasol_transformers_extension.udfs.models.ai_classify_extended_udf import (
    AiClassifyExtendedUDF,
)

udf = AiClassifyExtendedUDF(exa, work_with_spans=True)


def run(ctx):
    """
    run function for AiClassifyExtendedUDF
    """
    return udf.run(ctx)
