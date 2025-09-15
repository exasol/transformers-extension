from pathlib import Path

import exasol.bucketfs as bfs
import transformers as huggingface
from exasol.python_extension_common.connections.bucketfs_location import (
    create_bucketfs_location_from_conn_object,
)

from exasol_transformers_extension.utils import device_management
from exasol_transformers_extension.utils.bucketfs_model_specification import (
    BucketFSModelSpecification,
)
from exasol_transformers_extension.utils.huggingface_hub_bucketfs_model_transfer_sp import (
    HuggingFaceHubBucketFSModelTransferSP,
)
from exasol_transformers_extension.utils.load_local_model import LoadLocalModel
from exasol_transformers_extension.utils.model_factory_protocol import (
    ModelFactoryProtocol,
)


# Former name: download_transformers_model()
def install_huggingface_model(
    bucketfs_location: bfs.path.PathLike,
    sub_dir: str,
    task_type: str,
    model_name: str,
    model_factory: ModelFactoryProtocol,
    tokenizer_factory=huggingface.AutoTokenizer,
    huggingface_token: str | None = None,
) -> bfs.path.PathLike:
    """
    Downloads the specified model from the Huggingface hub into the BucketFS.
    Returns BucketFS location where the model is uploaded.

    Note: This function should NOT be called from a UDF.

    Parameters:
        bucketfs_location:
            Root location in the BucketFS.
        sub_dir:
            Root subdirectory in the BucketFS location where all models are uploaded.
        task_type:
            Name of an NLP task recognized by the huggingface.pipeline(). See
            https://huggingface.co/docs/transformers/v4.48.2/en/main_classes/pipelines#transformers.pipeline.task
        model_name:
            Name of the model. This is the same name as it's seen on the Haggingface
            model card, for example 'cross-encoder/nli-deberta-base'.
        model_factory:
            The model class (AutoModelXXX), e.g. AutoModelForTokenClassification.
        tokenizer_factory:
            The tokenizer class, e.g. huggingface.AutoTokenizer
        huggingface_token:
            Optional Huggingface token, required for downloading a private mode.
    """
    model_spec = BucketFSModelSpecification(model_name, task_type, "", Path(sub_dir))

    # Get model path in the BucketFS
    model_path = model_spec.get_bucketfs_model_save_path()

    # Download the model and the tokenizer into the model path
    with HuggingFaceHubBucketFSModelTransferSP(
        bucketfs_location=bucketfs_location,
        model_specification=model_spec,
        bucketfs_model_path=model_path,
        token=huggingface_token,
    ) as downloader:
        for factory in [model_factory, tokenizer_factory]:
            downloader.download_from_huggingface_hub(factory)
        upload_path = downloader.upload_to_bucketfs()
    return bucketfs_location / upload_path


def load_huggingface_pipeline(
    exa,
    bucketfs_conn_name: str,
    sub_dir: str,
    device: int,
    task_type: str,
    model_name: str,
    model_factory,
    tokenizer_factory=huggingface.AutoTokenizer,
) -> huggingface.Pipeline:
    """
    Loads the specified model and returns a huggingface.pipeline object.

    This function would normally be called from a UDF.

    Parameters:
        exa:
            UDF meta-data object.
        bucketfs_conn_name:
            Name of the BucketFS connection object with BucketFS connection credentials.
        sub_dir:
            The root subdirectory in the BucketFS where all model are uploaded.
        device:
            Device ordinal for CPU/GPU supports. Setting this to -1 (default) will
            leverage CPU, a positive value will run the model on the associated CUDA
            device id.
        task_type:
            Name of an NLP task recognized by the Huggingface pipeline(). See
            https://huggingface.co/docs/transformers/v4.48.2/en/main_classes/pipelines#transformers.pipeline.task
        model_name:
            Name of the model. This is the same name as it's seen on the Haggingface
            model card, for example 'cross-encoder/nli-deberta-base'.
        model_factory:
            The model class (AutoModelXXX), e.g. AutoModelForTokenClassification.
        tokenizer_factory:
            The tokenizer class, e.g. huggingface.AutoTokenizer
    """
    device_obj = device_management.get_torch_device(device)
    model_loader = LoadLocalModel(
        pipeline_factory=huggingface.pipeline,
        base_model_factory=model_factory,
        tokenizer_factory=tokenizer_factory,  # type: ignore
        task_type=task_type,
        device=device_obj,  # type: ignore
    )
    model_spec = BucketFSModelSpecification(
        model_name,
        task_type,
        bucketfs_conn_name,
        Path(sub_dir),
    )
    bucketfs_location = get_bucketfs_location(exa, bucketfs_conn_name)

    model_loader.clear_device_memory()
    model_loader.set_current_model_specification(model_spec)
    model_loader.set_bucketfs_model_cache_dir(bucketfs_location)
    return model_loader.load_models()


def get_bucketfs_location(exa, bucketfs_conn_name: str) -> bfs.path.PathLike:
    return create_bucketfs_location_from_conn_object(
        exa.get_connection(bucketfs_conn_name)
    )
