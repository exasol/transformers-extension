from exasol_transformers_extension.udfs.models.ai_entailment_extended_udf import (
    AiEntailmentExtendedUDF,
)

udf = AiEntailmentExtendedUDF(exa)


def run(ctx):
    return udf.run(ctx)
