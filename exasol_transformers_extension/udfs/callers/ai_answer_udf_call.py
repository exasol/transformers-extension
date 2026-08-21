"""
Caller for AiAnswerUDF
"""

from exasol_transformers_extension.udfs.models.ai_answer import AiAnswerUDF

udf = AiAnswerUDF(exa)


def run(ctx):
    """
    run function for AiAnswerUDF
    """
    return udf.run(ctx)
