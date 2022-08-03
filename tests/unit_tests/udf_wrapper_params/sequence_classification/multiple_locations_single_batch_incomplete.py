from tests.unit_tests.udf_wrapper_params.sequence_classification.\
    mock_sequence_classification_factory import \
    Config, MockSequenceClassificationFactory, MockSequenceClassificationModel


def udf_wrapper_single_text():
    from exasol_udf_mock_python.udf_context import UDFContext
    from exasol_transformers_extension.udfs. \
        sequence_classification_single_text_udf import \
        SequenceClassificationSingleText
    from tests.unit_tests.udf_wrapper_params.sequence_classification. \
        mock_sequence_tokenizer import MockSequenceTokenizer
    from tests.unit_tests.udf_wrapper_params.sequence_classification.\
        multiple_model_single_batch_complete import \
        MultipleModelSingleBatchComplete as params

    udf = SequenceClassificationSingleText(
        exa,
        batch_size=params.batch_size,
        base_model=params.mock_factory,
        tokenizer=MockSequenceTokenizer)

    def run(ctx: UDFContext):
        udf.run(ctx)


def udf_wrapper_text_pair():
    from exasol_udf_mock_python.udf_context import UDFContext
    from exasol_transformers_extension.udfs.\
        sequence_classification_text_pair_udf import \
        SequenceClassificationTextPair
    from tests.unit_tests.udf_wrapper_params.sequence_classification. \
        mock_sequence_tokenizer import MockSequenceTokenizer
    from tests.unit_tests.udf_wrapper_params.sequence_classification.\
        multiple_locations_single_batch_incomplete import \
        MultipleLocationsSingleBatchIncomplete as params

    udf = SequenceClassificationTextPair(
        exa,
        batch_size=params.batch_size,
        base_model=params.mock_factory,
        tokenizer=MockSequenceTokenizer)

    def run(ctx: UDFContext):
        udf.run(ctx)


class MultipleLocationsSingleBatchIncomplete:
    """
    multiple locations, single batch, last batch incomplete
    """

    batch_size = 9
    data_size = 2

    config = Config({
        0: 'label1', 1: 'label2',
        2: 'label3', 3: 'label4'})

    logits = [0.1, 0.2, 0.3, 0.4]

    mock_factory = MockSequenceClassificationFactory({
        "model1": MockSequenceClassificationModel(
            config=config,
            logits=logits),
        "model2": MockSequenceClassificationModel(
            config=config,
            logits=logits)
    })

    inputs_single_text = [(None, "bfs_conn1", "sub_dir1",
                           "model1", "My test text")] * data_size + \
                         [(None, "bfs_conn2", "sub_dir1",
                           "model1", "My test text")] * data_size + \
                         [(None, "bfs_conn1", "sub_dir2",
                           "model1", "My test text")] * data_size + \
                         [(None, "bfs_conn1", "sub_dir1",
                           "model2", "My test text")] * data_size
    inputs_pair_text = [(None, "bfs_conn1", "sub_dir1", "model1",
                         "My text 1", "My text 2")] * data_size + \
                       [(None, "bfs_conn2", "sub_dir1", "model1",
                         "My text 1", "My text 2")] * data_size + \
                       [(None, "bfs_conn1", "sub_dir2", "model1",
                         "My text 1", "My text 2")] * data_size + \
                       [(None, "bfs_conn1", "sub_dir1", "model2",
                         "My text 1", "My text 2")] * data_size

    outputs_single_text = [("bfs_conn1", "sub_dir1", "model1",
                            "My test text", "label1", 0.21),
                           ("bfs_conn1", "sub_dir1", "model1",
                            "My test text", "label2", 0.24),
                           ("bfs_conn1", "sub_dir1", "model1",
                            "My test text", "label3", 0.26),
                           ("bfs_conn1", "sub_dir1", "model1",
                            "My test text", "label4", 0.29)] * data_size + \
                          [("bfs_conn2", "sub_dir1", "model1",
                            "My test text", "label1", 0.21),
                           ("bfs_conn2", "sub_dir1", "model1",
                            "My test text", "label2", 0.24),
                           ("bfs_conn2", "sub_dir1", "model1",
                            "My test text", "label3", 0.26),
                           ("bfs_conn2", "sub_dir1", "model1",
                            "My test text", "label4", 0.29)] * data_size + \
                          [("bfs_conn1", "sub_dir2", "model1",
                            "My test text", "label1", 0.21),
                           ("bfs_conn1", "sub_dir2", "model1",
                            "My test text", "label2", 0.24),
                           ("bfs_conn1", "sub_dir2", "model1",
                            "My test text", "label3", 0.26),
                           ("bfs_conn1", "sub_dir2", "model1",
                            "My test text", "label4", 0.29)] * data_size + \
                          [("bfs_conn1", "sub_dir1", "model2",
                            "My test text", "label1", 0.21),
                           ("bfs_conn1", "sub_dir1", "model2",
                            "My test text", "label2", 0.24),
                           ("bfs_conn1", "sub_dir1", "model2",
                            "My test text", "label3", 0.26),
                           ("bfs_conn1", "sub_dir1", "model2",
                            "My test text", "label4", 0.29)] * data_size

    outputs_text_pair = [("bfs_conn1", "sub_dir1", "model1", "My text 1",
                          "My text 2", "label1", 0.21),
                         ("bfs_conn1", "sub_dir1", "model1", "My text 1",
                          "My text 2", "label2", 0.24),
                         ("bfs_conn1", "sub_dir1", "model1", "My text 1",
                          "My text 2", "label3", 0.26),
                         ("bfs_conn1", "sub_dir1", "model1", "My text 1",
                          "My text 2", "label4", 0.29)] * data_size + \
                        [("bfs_conn2", "sub_dir1", "model1", "My text 1",
                          "My text 2", "label1", 0.21),
                         ("bfs_conn2", "sub_dir1", "model1", "My text 1",
                          "My text 2", "label2", 0.24),
                         ("bfs_conn2", "sub_dir1", "model1", "My text 1",
                          "My text 2", "label3", 0.26),
                         ("bfs_conn2", "sub_dir1", "model1", "My text 1",
                          "My text 2", "label4", 0.29)] * data_size + \
                        [("bfs_conn1", "sub_dir2", "model1", "My text 1",
                          "My text 2", "label1", 0.21),
                         ("bfs_conn1", "sub_dir2", "model1", "My text 1",
                          "My text 2", "label2", 0.24),
                         ("bfs_conn1", "sub_dir2", "model1", "My text 1",
                          "My text 2", "label3", 0.26),
                         ("bfs_conn1", "sub_dir2", "model1", "My text 1",
                          "My text 2", "label4", 0.29)] * data_size + \
                        [("bfs_conn1", "sub_dir1", "model2", "My text 1",
                          "My text 2", "label1", 0.21),
                         ("bfs_conn1", "sub_dir1", "model2", "My text 1",
                          "My text 2", "label2", 0.24),
                         ("bfs_conn1", "sub_dir1", "model2", "My text 1",
                          "My text 2", "label3", 0.26),
                         ("bfs_conn1", "sub_dir1", "model2", "My text 1",
                          "My text 2", "label4", 0.29)] * data_size

    udf_wrapper_single_text = udf_wrapper_single_text
    udf_wrapper_text_pair = udf_wrapper_text_pair
