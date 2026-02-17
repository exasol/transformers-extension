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

    bucketfs_conn        | sub_dir                 | model_name                | task_type
    -----------------------------------------------------------------------------------------
    BucketFS connection | directory to save model | name of Huggingface model | type of model


    returns <bucketfs_conn>, <sub_dir>, <model_name>, <task_type>,
    <success>, <error_msg>
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
        bucketfs_conn, sub_dir, model_name, task_type = (
            ctx.bucketfs_conn,
            ctx.sub_dir,
            ctx.model_name,
            ctx.task_type,
        )

        try:
            current_model_specification = (
                self._current_model_specification_factory.create(
                    model_name, task_type, bucketfs_conn, Path(sub_dir)
                )
            )  # specifies details of Huggingface model
        except ValueError as e:
            # if task_type is not allowed for model_specification, use a placeholder for creation
            # and then replace using the legacy_set_task_type_from_udf_name.
            # needed to allow for deletion of already installed models with illegal task_types
            current_model_specification = (
                self._current_model_specification_factory.create(
                    model_name, "fill_mask", bucketfs_conn, Path(sub_dir)
                )
            )  # specifies details of Huggingface model
            current_model_specification.task_type = (
                current_model_specification.legacy_set_task_type_from_udf_name(
                    task_type
                )
            )
        try:
            # create bucketfs location
            bfs_conn_obj = self._exa.get_connection(bucketfs_conn)

            bucketfs_location = bfs_loc.create_bucketfs_location_from_conn_object(
                bfs_conn_obj
            )
            delete_model(bucketfs_location, current_model_specification)
        except Exception as e:
            return bucketfs_conn, sub_dir, model_name, task_type, False, str(e)

        return bucketfs_conn, sub_dir, model_name, task_type, True, ""
