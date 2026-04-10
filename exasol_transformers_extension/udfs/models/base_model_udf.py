from abc import (
    ABC,
)
import transformers

from exasol_transformers_extension.udfs.models.prediction_tasks.prediction_task import (
    PredictionTask,
)

from exasol_transformers_extension.udfs.models.transformation.transformation_pipeline import (
    TransformationPipeline,
)
from exasol_transformers_extension.utils import (
    device_management,
)
from exasol_transformers_extension.utils.load_local_model import LoadLocalModel
from exasol_transformers_extension.utils.model_factory_protocol import (
    ModelFactoryProtocol,
)


class BaseModelUDF(ABC):
    """
    This base class for prediction udfs. Calls transform function of the given Transformations in
    "transformations" in order.
    """

    def __init__(
        self,
        batch_size: int,
        pipeline: transformers.Pipeline,
        base_model: ModelFactoryProtocol,
        tokenizer: ModelFactoryProtocol,
        prediction_task: PredictionTask,
        transformations: TransformationPipeline,
    ):
        self.batch_size = batch_size
        self.pipeline = pipeline
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = None
        self.model_loader = None
        self.prediction_task = prediction_task
        self.transformations = transformations

    def run(self, ctx):
        """
        run function for prediction udfs. This is where the execution of the udf starts.
        Calls transform function of the given Transformations in "self.transformations"
        in order.
        Emits the resulting dataframes. Results might be split into many dataframes,
        depending on which Transformations are used.
        You can also change the input and output columns via the transformations.
        """
        device_id = ctx.get_dataframe(1).iloc[0]["device_id"]
        self.device = device_management.get_torch_device(device_id)
        self.create_model_loader()
        ctx.reset()

        while True:
            batch_df = ctx.get_dataframe(num_rows=self.batch_size, start_col=1)
            if batch_df is None:
                break

            output_generator = self.transformations.execute(batch_df, self.model_loader)

            for result_df in output_generator:
                ctx.emit(result_df)

        self.model_loader.clear_device_memory()

    def create_model_loader(self):
        """
        Creates the model_loader.
        """
        self.model_loader = LoadLocalModel(
            pipeline_factory=self.pipeline,
            base_model_factory=self.base_model,
            tokenizer_factory=self.tokenizer,
            task_type=self.prediction_task.task_type,
            device=self.device,
        )
