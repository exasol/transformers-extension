from exasol_transformers_extension.udfs.models.ls_models_udf import (
    ListModelsUDF,
)

udf = ListModelsUDF(exa)


def run(ctx):
    return udf.run(ctx)