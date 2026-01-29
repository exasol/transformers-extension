from exasol_transformers_extension.udfs.models.ai_extract_extended_udf import (
    AiExtractExtendedUDF,
)

udf = AiExtractExtendedUDF(exa, work_with_spans=True)


def run(ctx):
    return udf.run(ctx)
