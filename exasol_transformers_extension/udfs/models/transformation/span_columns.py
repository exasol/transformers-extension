from collections.abc import Iterator

from pandas import (
    DataFrame,
    Series,
)

from exasol_transformers_extension.udfs.models.transformation.transformation import (
    Transformation,
)
from exasol_transformers_extension.udfs.models.transformation.utils import (
    _check_input_format,
    _create_new_empty_columns,
    _drop_old_columns,
    _ensure_output_format,
)
from exasol_transformers_extension.utils.load_local_model import LoadLocalModel


class SpanColumnsTokenClassificationTransformation(Transformation):
    def __init__(
        self,
        expected_input_columns: list[str],
        new_columns: list[str],
        removed_columns: list[str],
    ):
        self.renamed_columns = {"word": "entity_covered_text", "entity": "entity_type"}
        self.expected_input_columns = expected_input_columns
        self.new_columns = new_columns
        self.removed_columns = removed_columns

    def rename_columns(self, model_df: DataFrame) -> DataFrame:
        # we use different names in udf with span and without, so need to rename
        # this decision was made as to improve the naming of the columns without
        # breaking the interface of the existing udf
        model_df = model_df.rename(columns=self.renamed_columns)
        return model_df

    @staticmethod
    def make_entity_span(df_row):
        token_doc_id = df_row["text_data_doc_id"]
        token_char_begin = df_row["start_pos"] + df_row["text_data_char_begin"]
        token_char_end = df_row["end_pos"] + df_row["text_data_char_begin"]
        return Series([token_doc_id, token_char_begin, token_char_end])

    def fill_span_columns(self, batch_df: DataFrame) -> DataFrame:
        batch_df[self.new_columns] = batch_df.apply(self.make_entity_span, axis=1)
        return batch_df

    def transform(
        self, batch_df: DataFrame, model_loader: LoadLocalModel
    ) -> list[DataFrame]:
        batch_df = _create_new_empty_columns(batch_df, self.new_columns)
        batch_df = self.rename_columns(batch_df)
        batch_df = self.fill_span_columns(batch_df)
        # drop columns which are made superfluous by the spans to save data transfer
        batch_df = _drop_old_columns(batch_df, self.removed_columns)
        return [batch_df]

    def check_input_format(self, df_columns: list[str]):
        """
        checks if all needed columns for
        transform are present, throws error otherwise
        """
        try:
            _check_input_format(
                df_columns, self.expected_input_columns, self.__class__.__name__
            )
        except Exception as e:
            raise e

    def ensure_output_format(self, batch_df: DataFrame) -> DataFrame:
        """
        ensure all promised output columns are present
        """
        batch_df = self.rename_columns(batch_df)
        return _ensure_output_format(batch_df, self.new_columns, self.removed_columns)


class SpanColumnsZeroShotTransformation(Transformation):
    def __init__(
        self,
        expected_input_columns: list[str],
        new_columns: list[str],
        removed_columns: list[str],
    ):
        self.expected_input_columns = expected_input_columns
        self.new_columns = new_columns
        self.removed_columns = removed_columns

    def transform(
        self, batch_df: DataFrame, model_loader: LoadLocalModel
    ) -> list[DataFrame]:
        batch_df = _create_new_empty_columns(batch_df, self.new_columns)
        # drop columns which are made superfluous by the spans to save data transfer
        batch_df = _drop_old_columns(batch_df, self.removed_columns)
        return [batch_df]

    def check_input_format(self, df_columns: list[str]):
        """
        checks if all needed columns for
        transform are present, throws error otherwise
        """
        try:
            _check_input_format(
                df_columns, self.expected_input_columns, self.__class__.__name__
            )
        except Exception as e:
            raise e

    def ensure_output_format(self, batch_df: DataFrame) -> DataFrame:
        """
        ensure all promised output columns are present
        """
        return _ensure_output_format(batch_df, self.new_columns, self.removed_columns)
