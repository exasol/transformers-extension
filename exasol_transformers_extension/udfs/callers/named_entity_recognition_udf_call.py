from exasol_transformers_extension.udfs.models.named_entity_recognition_udf import \
    NamedEntityRecognitionUDF

udf = NamedEntityRecognitionUDF(exa)


def run(ctx):
    return udf.run(ctx)
