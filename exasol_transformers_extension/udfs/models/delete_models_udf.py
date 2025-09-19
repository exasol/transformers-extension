from pathlib import Path

import exasol.python_extension_common.connections.bucketfs_location as bfs_loc

from exasol_transformers_extension.utils.bucketfs_model_specification import (
    BucketFSModelSpecificationFactory,
)
from exasol_transformers_extension.utils.model_utils import delete_model


class DeleteModelUDF:
    """
    UDF which deletes a pretrained model from BucketFS.
    Must be called with the following Input Parameter:

    model_name                | task_type              | sub_dir                 | bfs_conn
    ---------------------------------------------------------------------------------------------------
    name of Huggingface model | type of model          | directory to save model | BucketFS connection

    returns <model_name> , <task_type>, <sub_dir>, <bfs_conn>, <success>, <error_msg>
    """

    def __init__(
        self,
        exa,
        current_model_specification_factory: BucketFSModelSpecificationFactory = BucketFSModelSpecificationFactory(),
   ):
        self._exa = exa
        self._current_model_specification_factory = current_model_specification_factory

    def run(self, ctx) -> None:
        while True:
            model_delete_result = self._delete_model(ctx)
            ctx.emit(*model_delete_result)
            if not ctx.next():
                break


    def _delete_model(self, ctx) -> tuple[str, str, str, str, bool, str]:
        # parameters
        model_name, task_type, sub_dir, bfs_conn = ctx.model_name, ctx.task_type, ctx.sub_dir, ctx.bfs_conn

        current_model_specification = self._current_model_specification_factory.create(
            model_name, task_type, bfs_conn, Path(sub_dir)
        )  # specifies details of Huggingface model
        try:
            # create bucketfs location
            bfs_conn_obj = self._exa.get_connection(bfs_conn)

            bucketfs_location = bfs_loc.create_bucketfs_location_from_conn_object(
                bfs_conn_obj
            )
            delete_model(bucketfs_location, current_model_specification)
        except Exception as e:
            return model_name, task_type, sub_dir, bfs_conn, False, str(e)

        return model_name, task_type, sub_dir, bfs_conn, True, ""
