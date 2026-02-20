from exasol_transformers_extension.udfs.models.ai_classify_extended_udf import (
    AiClassifyExtendeUDF,
)

udf = AiClassifyExtendeUDF(exa, work_with_spans=True)


def run(ctx):
    return udf.run(ctx)
