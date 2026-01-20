from exasol_transformers_extension.udfs.models.ai_translate_extended_udf import AiTranslateExtendedUDF

udf = AiTranslateExtendedUDF(exa)


def run(ctx):
    return udf.run(ctx)
