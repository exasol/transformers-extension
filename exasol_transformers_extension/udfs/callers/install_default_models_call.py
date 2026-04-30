"""
Caller for InstallDefaultModelsUDF
"""

from exasol_transformers_extension.udfs.models.install_default_models_udf import (
    InstallDefaultModelsUDF,
)

udf = InstallDefaultModelsUDF(exa)


def run(ctx):
    """
    run function for InstallDefaultModelsUDF
    """
    return udf.run(ctx)
