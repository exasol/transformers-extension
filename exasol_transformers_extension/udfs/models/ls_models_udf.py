from typing import Tuple

import exasol.python_extension_common.connections.bucketfs_location as bfs_loc
import transformers

from exasol_transformers_extension.utils.bucketfs_model_specification import (
    BucketFSModelSpecificationFactory,
)
from exasol_transformers_extension.utils.huggingface_hub_bucketfs_model_transfer_sp import (
    HuggingFaceHubBucketFSModelTransferSPFactory,
)
from exasol_transformers_extension.utils.model_factory_protocol import (
    ModelFactoryProtocol,
)


class ListModelsUDF:
    """
    UDF which list all transformers models installed with the Transformers Extension in the BucketFS.
    Must be called with the following Input Parameter:

    | sub_dir                 | bfs_conn            |
    -------------------------------------------------
    | directory to save model | BucketFS connection |

    returns a table of  <sub_dir/model_name> , <path of model BucketFS>
    path, subdir, version, task_name, model_name ?

    udf bekommt bucketfs-connection

    function bekommt bucketfs-model-path?
    """

    def __init__(
        self,
        exa,
    ):
        self._exa = exa

    def run(self, ctx) -> None:
        model_path_list = self._list_models(ctx)
        for model in model_path_list:
            ctx.emit(*model)

    def _list_models(self, ctx):#todo type hints
        # parameters
        bfs_conn = ctx.bucketfs_conn  # BucketFS connection
        print(bfs_conn)
        #bfs_conn = ctx.get_dataframe(1).iloc[0]["bucketfs_conn"]
        sub_dir = ctx.sub_dir
        # create bucketfs location
        bfs_conn_obj = self._exa.get_connection(bfs_conn)

        from exasol.bucketfs import (
            MappedBucket,
            Service,
        )
        import json

        #URL = json.loads(bfs_conn_obj.address)
        #CREDENTIALS = {"default": {"username": json.loads(bfs_conn_obj.user), "password": json.loads(bfs_conn_obj.password)}}
        #bucketfs = Service(URL, CREDENTIALS)

        #default_bucket = MappedBucket(bucketfs["default"])
        #files = [file for file in default_bucket]
        #print(files)

        bucketfs_location = bfs_loc.create_bucketfs_location_from_conn_object(
            bfs_conn_obj
        )
        models_list = []
        #while True:
        if bucketfs_location.is_file():
             models_list.append(bucketfs_location.as_udf_path())
        elif bucketfs_location.is_dir():
             for item in bucketfs_location.iterdir():
                 models_list.append(bucketfs_location.as_udf_path())

        print(models_list)
        return models_list
