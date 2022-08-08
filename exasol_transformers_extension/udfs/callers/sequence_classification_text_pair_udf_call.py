from exasol_transformers_extension.udfs.models.sequence_classification_text_pair_udf \
    import SequenceClassificationTextPair

udf = SequenceClassificationTextPair(exa)


def run(ctx):
    return udf.run(ctx)
