from exasol_transformers_extension.udfs.models.ai_answer_extended_udf import (
    AiAnswerExtendedUDF,
)

udf = AiAnswerExtendedUDF(exa)


def run(ctx):
    return udf.run(ctx)
