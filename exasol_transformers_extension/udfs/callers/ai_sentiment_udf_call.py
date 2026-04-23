from exasol_transformers_extension.udfs.models.ai_sentiment_udf import AiSentimentUDF

udf = AiSentimentUDF(exa)


def run(ctx):
    return udf.run(ctx)
