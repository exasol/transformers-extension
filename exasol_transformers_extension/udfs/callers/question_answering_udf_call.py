from exasol_transformers_extension.udfs.models.question_answering_udf \
    import QuestionAnsweringUDF

udf = QuestionAnsweringUDF(exa)


def run(ctx):
    return udf.run(ctx)
