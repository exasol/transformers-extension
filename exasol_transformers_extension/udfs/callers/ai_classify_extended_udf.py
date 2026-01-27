from exasol_transformers_extension.udfs.models.zero_ai_classify_extended_udf import (
    AiClassifyExtendeUDF,
)

udf = AiClassifyExtendeUDF(exa)


def run(ctx):
    return udf.run(ctx)
