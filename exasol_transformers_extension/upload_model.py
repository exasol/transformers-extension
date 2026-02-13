"""Downloads model from Huggingface hub and the transfers model to database"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import transformers as huggingface
from exasol.python_extension_common.cli.std_options import (
    StdTags,
    make_option_secret,
    select_std_options,
)
from exasol.python_extension_common.connections.bucketfs_location import (
    create_bucketfs_location,
)

from exasol_transformers_extension.deploy import (
    TOKEN_ARG,
    get_opt_name,
)
from exasol_transformers_extension.utils.bucketfs_model_specification import (
    BucketFSModelSpecification,
)
from exasol_transformers_extension.utils.model_utils import install_huggingface_model

MODEL_NAME_ARG = "model_name"
TASK_TYPE_ARG = "task_type"
SUBDIR_ARG = "sub_dir"

LOG = logging.getLogger(__name__)

opt_token = {"type": str, "help": "Huggingface token for private models"}
make_option_secret(opt_token, prompt="Huggingface token")
opts = select_std_options([StdTags.BFS])
opts.append(
    click.Option(
        [get_opt_name(MODEL_NAME_ARG)],
        type=str,
        required=True,
        help="name of the model",
    )
)
opts.append(
    click.Option(
        [get_opt_name(TASK_TYPE_ARG)],
        type=str,
        required=True,
        help="the name of the task the model is used for",
    )
)
opts.append(
    click.Option(
        [get_opt_name(SUBDIR_ARG)],
        type=str,
        required=True,
        help="directory where the model is stored in the BucketFS",
    )
)

opts.append(click.Option([get_opt_name(TOKEN_ARG)], **opt_token))  # type: ignore


def upload_model(**kwargs) -> None:
    """
    Downloads model from Huggingface hub and the transfers model to database
    """
    bucketfs_location = create_bucketfs_location(**kwargs)
    spec = BucketFSModelSpecification(
        kwargs[MODEL_NAME_ARG],
        kwargs[TASK_TYPE_ARG],
        "",
        Path(kwargs[SUBDIR_ARG]),
    )
    upload_path = install_huggingface_model(
        bucketfs_location=bucketfs_location,
        model_spec=spec,
        tokenizer_factory=huggingface.AutoTokenizer,
        huggingface_token=kwargs[TOKEN_ARG],
    )
    print(
        "Your model or tokenizer has been saved in the BucketFS at: " + str(upload_path)
    )


upload_model_command = click.Command(None, params=opts, callback=upload_model)


if __name__ == "__main__":
    upload_model_command()
