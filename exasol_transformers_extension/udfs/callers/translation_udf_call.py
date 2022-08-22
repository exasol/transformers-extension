from exasol_transformers_extension.udfs.models.translation_udf import \
    TranslationUDF

udf = TranslationUDF(exa)


def run(ctx):
    return udf.run(ctx)
