from exasol_transformers_extension.udfs.models.token_classification_udf import (
    TokenClassificationUDF,
)

udf = TokenClassificationUDF(exa, work_with_spans=True)


def run(ctx):
    return udf.run(ctx)
