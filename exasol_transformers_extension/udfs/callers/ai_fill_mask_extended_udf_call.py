from exasol_transformers_extension.udfs.models.ai_fill_mask_extended_udf import AiFillMaskExtendedUDF

udf = AiFillMaskExtendedUDF(exa)


def run(ctx):
    return udf.run(ctx)
