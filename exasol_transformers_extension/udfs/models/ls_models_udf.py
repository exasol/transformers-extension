import os
import traceback
from pathlib import Path

import exasol.python_extension_common.connections.bucketfs_location as bfs_loc
from exasol.bucketfs._path import PathLike

from exasol_transformers_extension.utils import bucketfs_operations
from exasol_transformers_extension.utils.model_specification import create_model_specs_from_path


class ListModelsUDF:
    """
    UDF which list all transformers models installed with the Transformers Extension in the BucketFS/subdir.
    Must be called with the following Input Parameter:

    | sub_dir                     | bfs_conn            |
    -------------------------------------------------
    | directory where models are | BucketFS connection |

    returns a table of:
    bucketfs_conn, sub_dir, model_name, task_name, path of model in BucketFS
    """

    def __init__(
        self,
        exa,
    ):
        self._exa = exa
        self._output = []
        self._error_message = None

    def run(self, ctx) -> None:
        self._list_models(ctx)
        for model_info in self._output:
            ctx.emit(*model_info)

    def _search_modelpaths_in_dir(self, sub_dir: str, bucketfs_location: PathLike) -> list[str]:
        model_paths_list = []
        for main_dir, sub_dirs, files in os.walk(Path((bucketfs_location.as_udf_path() + "/" + sub_dir))):
            if files: #this means there is at least 1 file here
                for file in files:
                    # models saved with .from_pretrained can have different file types and directory structures,
                    # but always have a config.json
                    # https://huggingface.co/docs/diffusers/main/using-diffusers/other-formats
                    if file.endswith("config.json"):
                        # some models have multiple config files
                        # todo main_dir might be different for different config_files. take highest level one?
                        if not main_dir in model_paths_list:
                            model_paths_list.append(main_dir)
        return model_paths_list


    def _parse_model_info_from_path(self, model_paths_list: list[str], sub_dir: str, bfs_conn_name: str) -> None:
        for model_path in model_paths_list:
            try:
                model_spec = create_model_specs_from_path(Path(model_path), sub_dir)
                self._output.append([bfs_conn_name, sub_dir, model_spec.model_name, model_spec.task_type, model_path,
                               self._error_message])
            except Exception as exc:
                self._error_message = traceback.format_exc()
                self._output.append([bfs_conn_name, sub_dir, "", "", model_path, self._error_message])

    def _list_models(self, ctx):
        # parameters
        bfs_conn_name = ctx.bucketfs_conn  # BucketFS connection
        sub_dir = str(ctx.sub_dir)

        # create bucketfs location
        bfs_conn_obj = self._exa.get_connection(bfs_conn_name)
        bucketfs_location = bfs_loc.create_bucketfs_location_from_conn_object(
            bfs_conn_obj
        )

        if not sub_dir:
            self._error_message = "sub_dir cant be an empty string" #-> disallow "" this in creation of path at some point?
            self._output.append([bfs_conn_name, sub_dir, "", "", "", self._error_message])
            return self._output

        model_paths_list = self._search_modelpaths_in_dir(sub_dir, bucketfs_location)

        if not model_paths_list:
            self._error_message = "no models in this subdir" #todo do we want this message? or just return empty?
            self._output.append([bfs_conn_name, sub_dir, "", "", "", self._error_message])
            return self._output

        self._parse_model_info_from_path(model_paths_list, sub_dir, bfs_conn_name)
