from exasol_transformers_extension.udfs.sequence_classification_udf import \
    SequenceClassification

udf = SequenceClassification(exa)


def run(ctx):
    return udf.run(ctx)