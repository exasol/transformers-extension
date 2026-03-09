from exasol_transformers_extension.udfs.models.zero_ai_classify_extended_udf import (
    AiClassifyExtendedUDF,
)

udf = AiClassifyExtendedUDF(exa)


def run(ctx):
    return udf.run(ctx)
