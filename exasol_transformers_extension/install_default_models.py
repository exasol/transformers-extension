from exasol_transformers_extension.deployment.default_udf_parameters import DEFAULT_MODEL_SPECS
from exasol_transformers_extension.utils.model_utils import install_huggingface_model
import exasol.python_extension_common.connections.bucketfs_location as bfs_loc

import click
import transformers as huggingface

def install_default_models(**kwargs) -> None:
    """
    Downloads model from Huggingface hub and the transfers model to database
    """
    bfs_connection = DEFAULT_BFS_CONNECTION#-< todo this is a conn name, how to make connection?
    bfs_location = bfs_loc.create_bucketfs_location_from_conn_object(
        bfs_connection
    )
    default_models = DEFAULT_MODEL_SPECS
    #bucketfs_location = create_bucketfs_location(**kwargs)

    for model_spec in default_models:
        upload_path = install_huggingface_model(
            bucketfs_location=bfs_location,
            model_spec=model_spec,
            tokenizer_factory=huggingface.AutoTokenizer,
            huggingface_token=None,
        )
        print(
            "Your model or tokenizer has been saved in the BucketFS at: " + str(upload_path)
        )


install_default_models_command = click.Command(None, params=opts, callback=install_default_models())


if __name__ == "__main__":
    install_default_models_command ()