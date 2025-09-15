import transformers as huggingface

from exasol_transformers_extension.utils.model_specification import ModelSpecification
from exasol_transformers_extension.utils.model_utils import load_huggingface_pipline

DEVICE_CPU = -1


def run(ctx):
    mspec = ModelSpecification(ctx.model_name, ctx.task_type)
    p = load_huggingface_pipeline(
        ctx,
        bucketfs_conn_name=ctx.bfs_conn,
        sub_dir=ctx.sub_dir,
        device=DEVICE_CPU,
        task_type=ctx.task_type,
        model_name=ctx.model_name,
        model_factory=mspec.get_model_factory(),
        tokenizer_factory=huggingface.AutoTokenizer,
    )
    return p.task, p.framework, p.device
