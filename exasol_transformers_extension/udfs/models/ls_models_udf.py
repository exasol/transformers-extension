import os
from pathlib import Path

import exasol.python_extension_common.connections.bucketfs_location as bfs_loc

from exasol_transformers_extension.utils.model_specification import create_model_spcs_from_path


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
        for model_info in model_path_list:
            ctx.emit(*model_info)

    def _list_models(self, ctx):#todo type hints
        # parameters

        bfs_conn_name = ctx.bucketfs_conn  # BucketFS connection
        print("bfs_conn_name:")
        print(bfs_conn_name)
        #bfs_conn_name = ctx.get_dataframe(1).iloc[0]["bucketfs_conn"]
        sub_dir = ctx.sub_dir
        # create bucketfs location
        bfs_conn_obj = self._exa.get_connection(bfs_conn_name)
        print("bfs_conn_obj:")
        print(bfs_conn_obj)
        from exasol.bucketfs import (
            MappedBucket,
            Service,
        )
        import json

        #Address = json.loads(bfs_conn_obj.address)
        #URL = Address["base_path"]
        #CREDENTIALS = {"default": {"username": json.loads(bfs_conn_obj.user), "password": json.loads(bfs_conn_obj.password)}}
        #bucketfs = Service(URL, CREDENTIALS)#does not ork with file as mock

        #default_bucket = MappedBucket(bucketfs["default"])
        #files = [file for file in default_bucket]
        #print(files)

        bucketfs_location = bfs_loc.create_bucketfs_location_from_conn_object(
            bfs_conn_obj
        )

        #check_path = bucketfs_operations.get_local_bucketfs_path(
        #    bucketfs_location=bucketfs_location, model_path=str(sub_dir)
        #)
        model_paths_list = []
        for main_dir, sub_dirs, files in os.walk(Path(bucketfs_location.as_udf_path())):
            #todo this gives us all tar files in bucketfs location. do we want to restrict it to subdir?
            # todo maybe input has multiple subdirs?
            if files: #this means there is at least 1 file here
                for file in files:
                    #if file.endswith(".tar.gz"): #todo do they get unzipped?
                    model_paths_list.append(main_dir + "/"+ file)


        print(model_paths_list)#todo format this into something usefull
        output = []#todo we output absolute path. do we want relative?
        for model_path in model_paths_list:
            model_spec = create_model_spcs_from_path(Path(model_path), sub_dir)
            output.append([bfs_conn_name, sub_dir, model_spec.model_name, model_spec.task_type, model_path, self._error_message])
        return output
