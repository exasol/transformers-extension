from exasol_transformers_extension.udfs.models.filling_mask_udf import FillingMask


udf = FillingMask(exa)


def run(ctx):
    return udf.run(ctx)
