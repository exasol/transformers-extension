"""
Caller for AiAnswerExtendedUDF
"""

from exasol_transformers_extension.udfs.models.ai_answer_extended_udf import (
    AiAnswerExtendedUDF,
)

udf = AiAnswerExtendedUDF(exa)


def run(ctx):
    """
    run function for AiAnswerExtendedUDF
    """
    return udf.run(ctx)
