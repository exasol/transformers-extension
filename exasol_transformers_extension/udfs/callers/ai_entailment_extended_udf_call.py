"""
Caller for AiEntailmentExtendedUDF
"""

from exasol_transformers_extension.udfs.models.ai_entailment_extended_udf import (
    AiEntailmentExtendedUDF,
)

udf = AiEntailmentExtendedUDF(exa)


def run(ctx):
    """
    run function for AiEntailmentExtendedUDF
    """
    return udf.run(ctx)
