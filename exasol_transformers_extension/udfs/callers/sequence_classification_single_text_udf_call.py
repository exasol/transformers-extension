from exasol_transformers_extension.udfs.models.sequence_classification_single_text_udf \
    import SequenceClassificationSingleTextUDF

udf = SequenceClassificationSingleTextUDF(exa)


def run(ctx):
    return udf.run(ctx)
