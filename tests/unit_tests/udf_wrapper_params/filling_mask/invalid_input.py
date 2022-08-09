from pathlib import PurePosixPath
from exasol_udf_mock_python.connection import Connection
from tests.unit_tests.udf_wrapper_params.filling_mask.mock_filling_mask import \
    MockFillingMaskFactory, MockFillingMaskModel, MockPipeline


def udf_wrapper():
    from exasol_udf_mock_python.udf_context import UDFContext
    from exasol_transformers_extension.udfs.models.filling_mask_udf import \
        FillingMask
    from tests.unit_tests.udf_wrapper_params.filling_mask.mock_sequence_tokenizer \
        import MockSequenceTokenizer
    from tests.unit_tests.udf_wrapper_params.filling_mask.invalid_input import \
        InvalidInput as params

    udf = FillingMask(
        exa,
        batch_size=params.batch_size,
        pipeline=params.mock_pipeline,
        base_model=params.mock_factory,
        tokenizer=MockSequenceTokenizer)

    def run(ctx: UDFContext):
        udf.run(ctx)


class InvalidInput:
    """
    Single model with different top_k input
    """
    batch_size = 2
    input_data = [(None, "bfs_conn1", "sub_dir1", "model1",
                   "text <mask> 1", 1)] + \
                 [(None, "bfs_conn1", "sub_dir1", "model1",
                   "text <mask> 1", 2)]

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, "bfs_conn1")
    bfs_connections = {
        "bfs_conn1": Connection(address=f"file://{base_cache_dir1}")}

    mock_factory = MockFillingMaskFactory({
        PurePosixPath(base_cache_dir1, "sub_dir1", "model1"):
            MockFillingMaskModel(sequence="text valid 1", score=0.1)
    })

    mock_pipeline = MockPipeline
    udf_wrapper = udf_wrapper
