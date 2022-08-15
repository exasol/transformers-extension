from exasol_transformers_extension.udfs.models.text_generation_udf \
    import TextGeneration

udf = TextGeneration(exa)


def run(ctx):
    return udf.run(ctx)
