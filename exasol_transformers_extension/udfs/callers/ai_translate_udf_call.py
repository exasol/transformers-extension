"""
Caller for AiTranslateUDF
"""

from exasol_transformers_extension.udfs.models.ai_translate import AiTranslateUDF

udf = AiTranslateUDF(exa)


def run(ctx):
    """
    run function for AiTranslateUDF
    """
    return udf.run(ctx)
