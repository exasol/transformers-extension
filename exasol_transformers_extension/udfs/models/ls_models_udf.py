import os
from pathlib import Path

import exasol.python_extension_common.connections.bucketfs_location as bfs_loc

from exasol_transformers_extension.utils import bucketfs_operations
from exasol_transformers_extension.utils.model_specification import create_model_specs_from_path


class ListModelsUDF:
    """
    UDF which list all transformers models installed with the Transformers Extension in the BucketFS.
    Must be called with the following Input Parameter:

    | sub_dir                 | bfs_conn            |
    -------------------------------------------------
    | directory where models are | BucketFS connection |

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
        self._error_message = None#todo where to set

    def run(self, ctx) -> None:
        model_path_list = self._list_models(ctx)
        # todo allow multiple rows with multiple subdirs as input?
        for model_info in model_path_list:
            ctx.emit(*model_info)

    def _list_models(self, ctx):#todo type hints
        # parameters

        bfs_conn_name = ctx.bucketfs_conn  # BucketFS connection
        sub_dir = str(ctx.sub_dir)

        # create bucketfs location
        bfs_conn_obj = self._exa.get_connection(bfs_conn_name)
        print("bfs_conn_obj:")
        print(bfs_conn_obj)

        bucketfs_location = bfs_loc.create_bucketfs_location_from_conn_object(
            bfs_conn_obj
        )
        #check_path = bucketfs_operations.get_local_bucketfs_path(
        #    bucketfs_location=bucketfs_location, model_path=str(sub_dir))
        #)
        #todo check if subdir is "". -> dissallow this in creation of path?
        model_paths_list = []
        for main_dir, sub_dirs, files in os.walk(Path((bucketfs_location.as_udf_path() + "/" + sub_dir))):
            if files: #this means there is at least 1 file here
                for file in files:
                    if file.endswith("config.json"): #todo  model files can be safetensors, ckpt or bin, config.json always exists if saved with .from_pretrained? not sure if could be renamed
                    # https://huggingface.co/docs/diffusers/main/using-diffusers/other-formats
                    # some models have multiple config files
                    # todo main_dir might be different
                        if not main_dir in model_paths_list:
                            model_paths_list.append(main_dir)


        print(model_paths_list)
        output = []
        for model_path in model_paths_list:
            model_spec = create_model_specs_from_path(Path(model_path), sub_dir)
            output.append([bfs_conn_name, sub_dir, model_spec.model_name, model_spec.task_type, model_path, self._error_message])
        return output
