"""
Caller for ModelDownloaderUDF
"""

from exasol_transformers_extension.udfs.models.model_downloader_udf import (
    ModelDownloaderUDF,
)

udf = ModelDownloaderUDF(exa)


def run(ctx):
    """
    run function for ModelDownloaderUDF
    """
    return udf.run(ctx)
