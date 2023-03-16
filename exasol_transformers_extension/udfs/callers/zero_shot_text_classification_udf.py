from exasol_transformers_extension.udfs.models.zero_shot_text_classification_udf \
    import ZeroShotTextClassificationUDF

udf = ZeroShotTextClassificationUDF(exa)


def run(ctx):
    return udf.run(ctx)
