from exasol_transformers_extension.udfs.models.question_answering_udf \
    import QuestionAnswering

udf = QuestionAnswering(exa)


def run(ctx):
    return udf.run(ctx)
