import traceback
from collections.abc import Iterator
from os import PathLike

import exasol.python_extension_common.connections.bucketfs_location as bfs_loc
from pandas import DataFrame

from exasol_transformers_extension.deployment.default_udf_parameters import (
    DEFAULT_BUCKETFS_CONN_NAME,
)
from exasol_transformers_extension.udfs.models.transformation.prediction_task import (
    PredictionTaskTransformation,
)
from exasol_transformers_extension.udfs.models.transformation.transformation import (
    Transformation,
)
from exasol_transformers_extension.utils.bucketfs_model_specification import (
    BucketFSModelSpecification,
)
from exasol_transformers_extension.utils.load_local_model import LoadLocalModel


class WithModelTransformation(Transformation):
    """
    Wrapper for a Transformation which needs to load a model.
    Loads a model if needed, then calls _transformation.transform
    """

    def __init__(
        self,
        exa,
        transformation: PredictionTaskTransformation,
    ):
        self._transformation = transformation
        self.exa = exa

    def transform(
        self, model_df: DataFrame, model_loader: LoadLocalModel
    ) -> Iterator[DataFrame]:
        """
        loads a model if needed, then calls _transformation.transform
        """
        self.check_cache(
            model_df, self._transformation.prediction_task.task_type, model_loader
        )
        yield from self._transformation.transform(model_df, model_loader)

    def check_input_format(self, df_columns: list[str]) -> None:
        self._transformation.check_input_format(df_columns)

    def ensure_output_format(self, batch_df: DataFrame) -> DataFrame:
        return self._transformation.ensure_output_format(batch_df)

    def _load_model(
        self,
        model_loader: LoadLocalModel,
        bucketfs_conn: PathLike,
        current_model_specification: BucketFSModelSpecification,
        current_device_id: str = None,
    ):
        """
        load a model into the cache
        """
        bucketfs_location = bfs_loc.create_bucketfs_location_from_conn_object(
            bucketfs_conn
        )

        model_loader.clear_device_memory()
        model_loader.set_current_device(current_device_id)
        model_loader.set_current_model_specification(current_model_specification)
        model_loader.set_bucketfs_model_cache_dir(bucketfs_location)

        self._transformation.prediction_task.last_created_pipeline = (
            model_loader.load_models()
        )

    @staticmethod
    def _build_error_msg(bucketfs_conn_name: str) -> str:
        main_msg = (
            f"You can create the required BucketFS connection by using the 'deploy' command, "
            f"or manually by executing the following: \n "
            f"CREATE OR REPLACE  CONNECTION {bucketfs_conn_name}  \n "
            f"TO <bucktfs_address> \n "
            f"USER <bucketfs_user>  \n "
            f"IDENTIFIED BY <bucketfs_password> "
            f"If you cannot create this connection yourself, "
            f"ask your admin. \n"
        )

        if bucketfs_conn_name == DEFAULT_BUCKETFS_CONN_NAME:
            msg = (
                f"In order to use this UDF, a BucketFS Connection by the name {DEFAULT_BUCKETFS_CONN_NAME} "
                f"must be created in the Exasol Database. "
            )
        else:
            msg = (
                f"The given BucketFS connection by the name of {bucketfs_conn_name} does not exist. "
                f"Either use another connection, or create it in the Exasol Database. "
            )
        return msg + main_msg

    def check_cache(
        self, model_df: DataFrame, task_type: str, model_loader: LoadLocalModel
    ) -> None:
        """
        If the model for the given dataframe is not cached, it is loaded into
        the cache before performing the prediction.

        If the model should have been cached but failed, another attempt will be made.

        :param model_df: Unique model dataframe having same model_name,
        :param task_type: transformers task the model will be used for
        :param model_loader: LoadLocalModel instance used to load the model
        bucketfs_connection, and sub_dir
        """
        model_name = model_df["model_name"].iloc[0]
        bucketfs_conn_name = model_df["bucketfs_conn"].iloc[0]
        sub_dir = model_df["sub_dir"].iloc[0]
        current_model_specification = BucketFSModelSpecification(
            model_name, task_type, bucketfs_conn_name, sub_dir
        )

        if (
            model_loader.current_model_specification != current_model_specification
            or not model_loader.last_model_loaded_successfully
        ):
            try:
                bucketfs_conn = self.exa.get_connection(bucketfs_conn_name)
            except Exception as e:
                msg = self._build_error_msg(bucketfs_conn_name)
                raise ConnectionError(msg) from e

            # if
            # we need to load a different model
            # or if
            # the model should have been loaded for the previous batch but failed,
            # we try again
            current_device_id = model_df["device_id"].iloc[0]
            self._load_model(
                model_loader,
                bucketfs_conn,
                current_model_specification,
                current_device_id,
            )
