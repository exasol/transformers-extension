from exasol_transformers_extension.udfs.models.filling_mask_udf import FillingMaskUDF


udf = FillingMaskUDF(exa)


def run(ctx):
    return udf.run(ctx)
