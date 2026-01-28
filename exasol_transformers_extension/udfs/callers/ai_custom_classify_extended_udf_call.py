from exasol_transformers_extension.udfs.models.ai_custom_classify_extended_udf import (
    AiCustomClassifyUDF,
)

udf = AiCustomClassifyUDF(exa)


def run(ctx):
    return udf.run(ctx)
