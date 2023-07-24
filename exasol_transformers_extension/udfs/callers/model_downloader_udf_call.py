from exasol_transformers_extension.udfs.models.model_downloader_udf import \
    ModelDownloaderUDF

udf = ModelDownloaderUDF(exa)


def run(ctx):
    return udf.run(ctx)