from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import click
import transformers as huggingface
from exasol.bucketfs._path import PathLike
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
from exasol_transformers_extension.utils.huggingface_hub_bucketfs_model_transfer_sp import (
    HuggingFaceHubBucketFSModelTransferSP,
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


def upload_model_to_bfs_location(
    model_name: str,
    task_type: str,
    subdir: Path,
    bucketfs_location: PathLike,
    huggingface_token: str | None = None,
) -> Path:
    """
    Deprecated.

    Please use
    exasol_transformers_extension.utils.model_utils.install_huggingface_model()
    instead.

    Downloads model from Huggingface hub and the transfers model to
    database at bucketfs_location

    params:
        model_name: name of the model
        task_type: name of the task model is used for
        subdir: directory where the model will be stored in the BucketFS
        bucketfs_location: BucketFS location model will be uploaded to
        huggingface_token: Optional. Huggingface token for private models

    returns
        path model/tokenizer is saved at in the BucketFS
    """
    LOG.warning(
        "This function is deprecated. "
        "Please use exasol_transformers_extension.utils"
        ".model_utils.install_huggingface_model() instead."
    )
    # create BucketFSModelSpecification for model to be loaded
    current_model_spec = BucketFSModelSpecification(model_name, task_type, "", subdir)
    # upload the downloaded model files into bucketfs
    upload_path = current_model_spec.get_bucketfs_model_save_path()

    model_factory = current_model_spec.get_model_factory()

    downloader = HuggingFaceHubBucketFSModelTransferSP(
        bucketfs_location=bucketfs_location,
        model_specification=current_model_spec,
        bucketfs_model_path=upload_path,
        token=huggingface_token,
    )

    for model in [model_factory, huggingface.AutoTokenizer]:
        downloader.download_from_huggingface_hub(model)
    # upload model files to BucketFS
    model_tar_file_path = downloader.upload_to_bucketfs()
    return model_tar_file_path


upload_model_command = click.Command(None, params=opts, callback=upload_model)


if __name__ == "__main__":
    upload_model_command()
