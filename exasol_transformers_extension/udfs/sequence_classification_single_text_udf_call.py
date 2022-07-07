from exasol_transformers_extension.udfs.sequence_classification_single_text_udf \
    import SequenceClassificationSingleText

udf = SequenceClassificationSingleText(exa)


def run(ctx):
    return udf.run(ctx)