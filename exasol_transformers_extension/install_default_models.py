import click
import transformers as huggingface
from exasol.python_extension_common.cli.std_options import (
    StdTags,
    select_std_options,
)
from exasol.python_extension_common.connections.bucketfs_location import (
    create_bucketfs_location,
)

from exasol_transformers_extension.deployment.default_udf_parameters import (
    DEFAULT_MODEL_SPECS,
)
from exasol_transformers_extension.utils.model_utils import install_huggingface_model

opts = select_std_options([StdTags.BFS])


def install_default_models(**kwargs) -> None:
    """
    Downloads default models from Huggingface hub and the transfers model to database
    """
    default_models = DEFAULT_MODEL_SPECS
    bucketfs_location = create_bucketfs_location(**kwargs)

    for model_spec in default_models:
        upload_path = install_huggingface_model(
            bucketfs_location=bucketfs_location,
            model_spec=model_spec,
            tokenizer_factory=huggingface.AutoTokenizer,
            huggingface_token=None,
        )
        print(
            "A model or tokenizer has been saved in the BucketFS at: "
            + str(upload_path)
        )


install_default_models_command = click.Command(
    None, params=opts, callback=install_default_models
)


if __name__ == "__main__":
    install_default_models_command()
