from exasol_transformers_extension.udfs.models.named_entity_recognition import \
    NamedEntityRecognitionUDF

udf = NamedEntityRecognitionUDF(exa)


def run(ctx):
    return udf.run(ctx)
