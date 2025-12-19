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


def install_huggingface_model(
    bucketfs_location: bfs.path.PathLike,
    model_spec: BucketFSModelSpecification,
    model_factory: ModelFactoryProtocol | None = None,
    tokenizer_factory=huggingface.AutoTokenizer,
    huggingface_token: str | None = None,
) -> bfs.path.PathLike:
    """
    Downloads the specified model from the Huggingface hub and uploads it
    into the BucketFS.  Returns the BucketFS location where the model is
    uploaded.

    Note: This function should NOT be called from a UDF.

    Parameters:
        bucketfs_location:
            Root location in the BucketFS.
        model_spec:
            BucketFSModelSpecification containing the model name, task type,
            and the subdirectory in the BucketFS for uploading the model to.
            Also provides the default model factory derived from the task type
            in case argument model_factory is None.
        model_factory:
            Optional model class (AutoModelXXX), e.g. AutoModelForTokenClassification.
            If set to None, the model factory is derived from the task-type
            in the model_spec.
        tokenizer_factory:
            Optional tokenizer class, e.g. huggingface.AutoTokenizer.
        huggingface_token:
            Optional Huggingface token, required for downloading a private mode.
    """
    # Get model path in the BucketFS
    model_path = model_spec.get_bucketfs_model_save_path()
    model_factory = model_factory or model_spec.get_model_factory()

    # Download the model and the tokenizer into the model path
    with HuggingFaceHubBucketFSModelTransferSP(
        bucketfs_location=bucketfs_location,
        model_specification=model_spec,
        bucketfs_model_path=model_path,
        token=huggingface_token,
    ) as installer:
        for factory in [model_factory, tokenizer_factory]:
            installer.download_from_huggingface_hub(factory)
        upload_path = installer.upload_to_bucketfs()
    return bucketfs_location / upload_path


def load_huggingface_pipeline(
    exa,
    model_spec: BucketFSModelSpecification,
    device: int,
    model_factory: ModelFactoryProtocol | None = None,
    tokenizer_factory=huggingface.AutoTokenizer,
) -> huggingface.Pipeline:
    """
    Loads the specified model and returns a huggingface.pipeline object.

    This function would normally be called from a UDF.

    Parameters:
        exa:
            UDF meta-data object.
        model_spec:
            BucketFSModelSpecification containing the model name, task type,
            the name of the BucketFS connection object with BucketFS
            connection credentials, and the subdirectory in the BucketFS where
            the model can be found at. Also provides the default model factory
            derived from the task type in case argument model_factory is None.
        device:
            Device ordinal for CPU/GPU supports. Setting this to -1 (default) will
            leverage CPU, a positive value will run the model on the associated CUDA
            device id.
        model_factory:
            Optional model class (AutoModelXXX), e.g. AutoModelForTokenClassification.
            If set to None, the model factory is derived from the task-type
            in the model_spec.
        tokenizer_factory:
            Optional tokenizer class, e.g. huggingface.AutoTokenizer
    """
    device_obj = device_management.get_torch_device(device)
    model_factory = model_factory or model_spec.get_model_factory()
    model_loader = LoadLocalModel(
        pipeline_factory=huggingface.pipeline,
        base_model_factory=model_factory,
        tokenizer_factory=tokenizer_factory,  # type: ignore
        task_type=model_spec.task_type,
        device=device_obj,  # type: ignore
    )
    bucketfs_location = get_bucketfs_location(exa, model_spec.bucketfs_conn_name)
    model_loader.clear_device_memory()
    model_loader.set_current_model_specification(model_spec)
    model_loader.set_bucketfs_model_cache_dir(bucketfs_location)
    return model_loader.load_models()


def get_bucketfs_location(exa, bucketfs_conn_name: str) -> bfs.path.PathLike:
    return create_bucketfs_location_from_conn_object(
        exa.get_connection(bucketfs_conn_name)
    )


def delete_model(
    bucketfs_location: bfs.path.PathLike,
    model_spec: BucketFSModelSpecification,
):
    """
    Deletes the specified model from BucketFS.

    Parameters:
        bucketfs_location:
            Root location in the BucketFS.
        model_spec:
            BucketFSModelSpecification containing the model name, task type,
            and the subdirectory in the BucketFS the model is saved at.
            Also provides the default model factory derived from the task type
            in case argument model_factory is None.
    """
    model_path = model_spec.get_bucketfs_model_save_path()

    delete_path = bucketfs_location / model_path.with_suffix(".tar.gz")

    delete_path.rm()
