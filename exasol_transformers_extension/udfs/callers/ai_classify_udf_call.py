"""Caller for AiClassifyUDF"""
from exasol_transformers_extension.udfs.models.ai_classify_udf import AiClassifyUDF

udf = AiClassifyUDF(exa)


def run(ctx):
    """run function for AiClassifyUDF"""
    return udf.run(ctx)
