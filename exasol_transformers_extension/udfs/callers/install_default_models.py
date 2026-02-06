from exasol_transformers_extension.udfs.models.install_default_models_udf import (
    InstallDefaultModelsUDF,
)

udf = InstallDefaultModelsUDF(exa)


def run(ctx):
    return udf.run(ctx)
