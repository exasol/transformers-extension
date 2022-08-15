from exasol_transformers_extension.udfs.models.text_generation_udf \
    import TextGenerationUDF

udf = TextGenerationUDF(exa)


def run(ctx):
    return udf.run(ctx)
