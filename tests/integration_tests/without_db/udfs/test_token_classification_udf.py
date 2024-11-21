from typing import Dict

import pandas as pd
import pytest
import torch
from exasol_udf_mock_python.connection import Connection

from exasol_transformers_extension.udfs.models.token_classification_udf import \
    TokenClassificationUDF
from exasol_transformers_extension.utils.model_specification import ModelSpecification
from tests.fixtures.model_fixture_utils import prepare_model_for_local_bucketfs

from tests.integration_tests.without_db.udfs.matcher import Result, ShapeMatcher, NewColumnsEmptyMatcher, \
    ErrorMessageMatcher, NoErrorMessageMatcher, ColumnsMatcher
from tests.utils.mock_connections import create_mounted_bucketfs_connection
from tests.utils.parameters import model_params


class ExaEnvironment:
    def __init__(self, connections: Dict[str, Connection] = None):
        self._connections = connections
        if self._connections is None:
            self._connections = {}

    def get_connection(self, name: str) -> Connection:
        return self._connections[name]


class Context:
    def __init__(self, input_df):
        self.input_df = input_df
        self._emitted = []
        self._is_accessed_once = False

    def emit(self, *args):
        self._emitted.append(args)

    def reset(self):
        self._is_accessed_once = False

    def get_emitted(self):
        return self._emitted

    def get_dataframe(self, num_rows='all', start_col=0):
        return_df = None if self._is_accessed_once \
            else self.input_df[self.input_df.columns[start_col:]]
        self._is_accessed_once = True
        return return_df


@pytest.mark.parametrize(
    "description,  device_id, n_rows, agg", [
        ("on CPU with batch input with none aggregation", None, 3, "none"),
        ("on CPU with batch input with NULL aggregation", None, 3, None),
        ("on CPU with batch input with max aggregation", None, 3, "max"),
        ("on CPU with single input with none aggregation", None, 1, "none"),
        ("on CPU with single input with NULL aggregation", None, 1, None),
        ("on CPU with single input with max aggregation", None, 1, "max"),
        ("on GPU with batch input with none aggregation", 0, 3, "none"),
        ("on GPU with batch input with NULL aggregation", 0, 3, None),
        ("on GPU with batch input with max aggregation", 0, 3, "max"),
        ("on GPU with single input with none aggregation", 0, 1, "none"),
        ("on GPU with single input with NULL aggregation", 0, 1, None),
        ("on GPU with single input with max aggregation", 0, 1, "max")
    ])
def test_token_classification_udf(
        description, device_id, n_rows, agg,
        prepare_token_classification_model_for_local_bucketfs):
    if device_id is not None and not torch.cuda.is_available():
        pytest.skip(f"There is no available device({device_id}) "
                    f"to execute the test")

    bucketfs_base_path = prepare_token_classification_model_for_local_bucketfs
    bucketfs_conn_name = "bucketfs_connection"
    bucketfs_connection = create_mounted_bucketfs_connection(bucketfs_base_path)

    batch_size = 2
    sample_data = [(
        None,
        bucketfs_conn_name,
        model_params.sub_dir,
        model_params.token_model_specs.model_name,
        model_params.text_data * (i + 1),
        agg
    ) for i in range(n_rows)]
    columns = [
        'device_id',
        'bucketfs_conn',
        'sub_dir',
        'model_name',
        'text_data',
        'aggregation_strategy'
    ]

    sample_df = pd.DataFrame(data=sample_data, columns=columns)
    ctx = Context(input_df=sample_df)
    exa = ExaEnvironment({bucketfs_conn_name: bucketfs_connection})

    sequence_classifier = TokenClassificationUDF(exa, batch_size=batch_size)
    sequence_classifier.run(ctx)

    result_df = ctx.get_emitted()[0][0]
    new_columns = \
        ['start_pos', 'end_pos', 'word', 'entity', 'score', 'error_message']

    result = Result(result_df)
    assert (
            result == ColumnsMatcher(columns=columns[1:], new_columns=new_columns)
            and result == NoErrorMessageMatcher()
    )



@pytest.mark.parametrize(
    "description,  device_id, n_rows, agg", [
        ("on CPU with batch input with none aggregation", None, 3, "none"),
        ("on CPU with batch input with NULL aggregation", None, 3, None),
        ("on CPU with batch input with max aggregation", None, 3, "max"),
        ("on CPU with single input with none aggregation", None, 1, "none"),
        ("on CPU with single input with NULL aggregation", None, 1, None),
        ("on CPU with single input with max aggregation", None, 1, "max"),
        ("on GPU with batch input with none aggregation", 0, 3, "none"),
        ("on GPU with batch input with NULL aggregation", 0, 3, None),
        ("on GPU with batch input with max aggregation", 0, 3, "max"),
        ("on GPU with single input with none aggregation", 0, 1, "none"),
        ("on GPU with single input with NULL aggregation", 0, 1, None),
        ("on GPU with single input with max aggregation", 0, 1, "max")
    ])
def test_token_classification_udf_with_span(
        description, device_id, n_rows, agg,tmpdir_factory,
        prepare_token_classification_model_for_local_bucketfs):
    if device_id is not None and not torch.cuda.is_available():
        pytest.skip(f"There is no available device({device_id}) "
                    f"to execute the test")

    model_spec = ModelSpecification("guishe/nuner-v2_fewnerd_fine_super", "token-classification")
    bucketfs_path = prepare_model_for_local_bucketfs(model_spec, tmpdir_factory)

    bucketfs_base_path = prepare_token_classification_model_for_local_bucketfs
    bucketfs_conn_name = "bucketfs_connection"
    bucketfs_connection = create_mounted_bucketfs_connection(bucketfs_path)
    text_data = "Foreign governments may be spying on your smartphone notifications, senator says. Washington (CNN) — Foreign governments have reportedly attempted to spy on iPhone and Android users through the mobile app notifications they receive on their smartphones - and the US government has forced Apple and Google to keep quiet about it, according to a top US senator. Through legal demands sent to the tech giants, governments have allegedly tried to force Apple and Google to turn over sensitive information that could include the contents of a notification - such as previews of a text message displayed on a lock screen, or an update about app activity, Oregon Democratic Sen. Ron Wyden said in a new report. Wyden''s report reflects the latest example of long-running tensions between tech companies and governments over law enforcement demands, which have stretched on for more than a decade. Governments around the world have particularly battled with tech companies over encryption, which provides critical protections to users and businesses while in some cases preventing law enforcement from pursuing investigations into messages sent over the internet.'"
    text_data2 = "This is a test."
    batch_size = 2
    sample_data = [(
        None,
        bucketfs_conn_name,
        model_params.sub_dir,
        'guishe/nuner-v2_fewnerd_fine_super',
        text_data2,
        i,
        0,
        len(model_params.text_data),
        "simple"
    ) for i in range(n_rows)]
    columns = [
        'device_id',
        'bucketfs_conn',
        'sub_dir',
        'model_name',
        'text_data',
        "text_data_docid",
        "text_data_char_begin",
        "text_data_char_end",
        'aggregation_strategy'
    ]

    old_columns = [
        'bucketfs_conn',
        'sub_dir',
        'model_name',
        "text_data_docid",
        "text_data_char_begin",
        "text_data_char_end",
        'aggregation_strategy'
    ]

    sample_df = pd.DataFrame(data=sample_data, columns=columns)
    ctx = Context(input_df=sample_df)
    exa = ExaEnvironment({bucketfs_conn_name: bucketfs_connection})

    sequence_classifier = TokenClassificationUDF(exa, batch_size=batch_size, work_with_spans=True)
    sequence_classifier.run(ctx)

    result_df = ctx.get_emitted()[0][0]
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also

        print(result_df)
    new_columns = \
        ['entity_covered_text', 'entity_type', 'score', 'entity_docid',
         'entity_char_begin', 'entity_char_end', 'error_message']

    result = Result(result_df)
    assert (False and
            result == ColumnsMatcher(columns=old_columns, new_columns=new_columns)
            and result == NoErrorMessageMatcher()
    )


@pytest.mark.parametrize(
    "description,  device_id", [
        ("on CPU", None),
        ("on GPU", 0)
    ])
def test_token_classification_udf_with_multiple_aggregation_strategies(
        description, device_id, prepare_token_classification_model_for_local_bucketfs):
    if device_id is not None and not torch.cuda.is_available():
        pytest.skip(f"There is no available device({device_id}) "
                    f"to execute the test")

    bucketfs_base_path = prepare_token_classification_model_for_local_bucketfs
    bucketfs_conn_name = "bucketfs_connection"
    bucketfs_connection = create_mounted_bucketfs_connection(bucketfs_base_path)

    batch_size = 2
    agg_strategies = [None, 'none', 'simple', 'max', 'average']
    sample_data = [(
        None,
        bucketfs_conn_name,
        model_params.sub_dir,
        model_params.token_model_specs.model_name,
        model_params.text_data * (i + 1),
        agg_strategy
    ) for i, agg_strategy in enumerate(agg_strategies)]
    columns = [
        'device_id',
        'bucketfs_conn',
        'sub_dir',
        'guishe/nuner-v2_fewnerd_fine_super',
        'text_data',
        'aggregation_strategy'
    ]

    sample_df = pd.DataFrame(data=sample_data, columns=columns)
    ctx = Context(input_df=sample_df)
    exa = ExaEnvironment({bucketfs_conn_name: bucketfs_connection})

    sequence_classifier = TokenClassificationUDF(exa, batch_size=batch_size)
    sequence_classifier.run(ctx)

    result_df = ctx.get_emitted()[0][0]
    new_columns = \
        ['start_pos', 'end_pos', 'word', 'entity', 'score', 'error_message']

    result = Result(result_df)
    assert (result == ColumnsMatcher(columns=columns[1:], new_columns=new_columns)
            and result == NoErrorMessageMatcher()
            and set(result_df['aggregation_strategy'].unique()) == {"none", "simple", "max", "average"}
            )


@pytest.mark.parametrize(
    "description,  device_id, n_rows, agg", [
        ("on CPU with batch input with none aggregation", None, 3, "none"),
        ("on CPU with batch input with NULL aggregation", None, 3, None),
        ("on CPU with batch input with max aggregation", None, 3, "max"),
        ("on CPU with single input with none aggregation", None, 1, "none"),
        ("on CPU with single input with NULL aggregation", None, 1, None),
        ("on CPU with single input with max aggregation", None, 1, "max"),
        ("on GPU with batch input with none aggregation", 0, 3, "none"),
        ("on GPU with batch input with NULL aggregation", 0, 3, None),
        ("on GPU with batch input with max aggregation", 0, 3, "max"),
        ("on GPU with single input with none aggregation", 0, 1, "none"),
        ("on GPU with single input with NULL aggregation", 0, 1, None),
        ("on GPU with single input with max aggregation", 0, 1, "max")
    ])
def test_token_classification_udf_on_error_handling(
        description, device_id, n_rows, agg,
        prepare_token_classification_model_for_local_bucketfs):
    if device_id is not None and not torch.cuda.is_available():
        pytest.skip(f"There is no available device({device_id}) "
                    f"to execute the test")

    bucketfs_base_path = prepare_token_classification_model_for_local_bucketfs
    bucketfs_conn_name = "bucketfs_connection"
    bucketfs_connection = create_mounted_bucketfs_connection(bucketfs_base_path)

    batch_size = 2
    sample_data = [(
        None,
        bucketfs_conn_name,
        model_params.sub_dir,
        "not existing model",
        model_params.text_data * (i + 1),
        agg
    ) for i in range(n_rows)]
    columns = [
        'device_id',
        'bucketfs_conn',
        'sub_dir',
        'model_name',
        'text_data',
        'aggregation_strategy'
    ]

    sample_df = pd.DataFrame(data=sample_data, columns=columns)
    ctx = Context(input_df=sample_df)
    exa = ExaEnvironment({bucketfs_conn_name: bucketfs_connection})

    sequence_classifier = TokenClassificationUDF(exa, batch_size=batch_size)
    sequence_classifier.run(ctx)

    result_df = ctx.get_emitted()[0][0]
    new_columns = \
        ['start_pos', 'end_pos', 'word', 'entity', 'score', 'error_message']

    result = Result(result_df)
    assert (
            result == ShapeMatcher(columns=columns, new_columns=new_columns, n_rows=n_rows)
            and result == ColumnsMatcher(columns=columns[1:], new_columns=new_columns)
            and result == NewColumnsEmptyMatcher(new_columns=new_columns)
            and result == ErrorMessageMatcher()
    )
