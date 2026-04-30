"""
Caller for AiCompleteExtendedUDF
"""

from exasol_transformers_extension.udfs.models.ai_complete_extended_udf import (
    AiCompleteExtendedUDF,
)

udf = AiCompleteExtendedUDF(exa)


def run(ctx):
    """
    run function for AiCompleteExtendedUDF
    """
    return udf.run(ctx)
