from exasol_transformers_extension.udfs.models.delete_models_udf import (
    DeleteModelUDF,
)

udf = DeleteModelUDF(exa)


def run(ctx):
    return udf.run(ctx)
