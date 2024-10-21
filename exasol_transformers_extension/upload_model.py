from __future__ import annotations
from pathlib import Path

import click
import transformers

from exasol.python_extension_common.cli.std_options import (
    StdTags, select_std_options, make_option_secret)
from exasol.python_extension_common.connections.bucketfs_location import create_bucketfs_location
from exasol_transformers_extension.deploy import get_opt_name, TOKEN_ARG
from exasol_transformers_extension.utils.bucketfs_model_specification import BucketFSModelSpecification
from exasol_transformers_extension.utils.huggingface_hub_bucketfs_model_transfer_sp import \
    HuggingFaceHubBucketFSModelTransferSP

MODEL_NAME_ARG = 'model_name'
TASK_TYPE_ARG = 'task_type'
SUBDIR_ARG = 'sub_dir'

opt_token = {'type': str, 'help': 'Huggingface token for private models'}
make_option_secret(opt_token, prompt='Huggingface token')
opts = select_std_options([StdTags.BFS])
opts.append(click.Option([get_opt_name(MODEL_NAME_ARG)], type=str, required=True,
                         help="name of the model"))
opts.append(click.Option([get_opt_name(TASK_TYPE_ARG)], type=str, required=True,
                         help="the name of the task the model is used for"))
opts.append(click.Option([get_opt_name(SUBDIR_ARG)], type=str, required=True,
                         help="directory where the model is stored in the BucketFS"))
opts.append(click.Option([get_opt_name(TOKEN_ARG)], **opt_token))


def upload_model(**kwargs) -> None:
    """
    Downloads model from Huggingface hub and the transfers model to database
    """
    # create BucketFSModelSpecification for model to be loaded
    current_model_spec = BucketFSModelSpecification(kwargs[MODEL_NAME_ARG], kwargs[TASK_TYPE_ARG],
                                                    "", Path(kwargs[SUBDIR_ARG]))
    # upload the downloaded model files into bucketfs
    upload_path = current_model_spec.get_bucketfs_model_save_path()

    # create bucketfs location
    bucketfs_location = create_bucketfs_location(**kwargs)

    model_factory = current_model_spec.get_model_factory()

    downloader = HuggingFaceHubBucketFSModelTransferSP(bucketfs_location=bucketfs_location,
                                                       model_specification=current_model_spec,
                                                       bucketfs_model_path=upload_path,
                                                       token=kwargs[TOKEN_ARG])

    for model in [model_factory, transformers.AutoTokenizer]:
        downloader.download_from_huggingface_hub(model)
        # upload model files to BucketFS
    model_tar_file_path = downloader.upload_to_bucketfs()
    print("Your model or tokenizer has been saved in the BucketFS at: " + str(model_tar_file_path))


upload_model_command = click.Command(None, params=opts, callback=upload_model)


if __name__ == '__main__':
    upload_model_command()
