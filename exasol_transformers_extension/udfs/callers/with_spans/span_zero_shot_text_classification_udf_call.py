from exasol_transformers_extension.udfs.models.zero_shot_text_classification_udf import \
    ZeroShotTextClassificationUDF

udf = ZeroShotTextClassificationUDF(exa, work_with_spans=True)


def run(ctx):
    return udf.run(ctx)