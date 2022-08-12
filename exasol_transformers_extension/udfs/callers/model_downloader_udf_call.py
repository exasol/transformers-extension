from exasol_transformers_extension.udfs.models.model_downloader_udf import \
    ModelDownloader

udf = ModelDownloader(exa)


def run(ctx):
    return udf.run(ctx)