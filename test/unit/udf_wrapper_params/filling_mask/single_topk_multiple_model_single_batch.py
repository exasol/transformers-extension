from pathlib import PurePosixPath
from test.unit.udf_wrapper_params.filling_mask.mock_filling_mask import (
    MockFillingMaskFactory,
    MockFillingMaskModel,
    MockPipeline,
)

from exasol_udf_mock_python.connection import Connection


def udf_wrapper():
    from test.unit.udf_wrapper_params.filling_mask.mock_sequence_tokenizer import (
        MockSequenceTokenizer,
    )
    from test.unit.udf_wrapper_params.filling_mask.single_topk_multiple_model_single_batch import (
        SingleTopkMultipleModelNameSingleBatch as params,
    )

    from exasol_udf_mock_python.udf_context import UDFContext

    from exasol_transformers_extension.udfs.models.filling_mask_udf import (
        FillingMaskUDF,
    )

    udf = FillingMaskUDF(
        exa,
        batch_size=params.batch_size,
        pipeline=params.mock_pipeline,
        base_model=params.mock_factory,
        tokenizer=MockSequenceTokenizer,
    )

    def run(ctx: UDFContext):
        udf.run(ctx)


class SingleTopkMultipleModelNameSingleBatch:
    """
    single topk, multiple model, single batch
    """

    expected_model_counter = 2
    batch_size = 4
    data_size = 2
    top_k = 3

    input_data = [
        (None, "bfs_conn1", "sub_dir1", "model1", "text <mask> 1", top_k)
    ] * data_size + [
        (None, "bfs_conn1", "sub_dir1", "model2", "text <mask> 2", top_k)
    ] * data_size
    output_data = [
        (
            "bfs_conn1",
            "sub_dir1",
            "model1",
            "text <mask> 1",
            top_k,
            "text valid 1",
            0.1,
            1,
            None,
        )
    ] * data_size * top_k + [
        (
            "bfs_conn1",
            "sub_dir1",
            "model2",
            "text <mask> 2",
            top_k,
            "text valid 2",
            0.2,
            1,
            None,
        )
    ] * data_size * top_k

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, "bfs_conn1")
    bfs_connections = {"bfs_conn1": Connection(address=f"file://{base_cache_dir1}")}

    mock_factory = MockFillingMaskFactory(
        {
            PurePosixPath(
                base_cache_dir1, "sub_dir1", "model1_fill-mask"
            ): MockFillingMaskModel(sequence="text valid 1", score=0.1, rank=1),
            PurePosixPath(
                base_cache_dir1, "sub_dir1", "model2_fill-mask"
            ): MockFillingMaskModel(sequence="text valid 2", score=0.2, rank=1),
        }
    )

    mock_pipeline = MockPipeline
    udf_wrapper = udf_wrapper
