from pandas import DataFrame, Series

from exasol_transformers_extension.udfs.models.transformation.transformation import Transformation


def _create_new_span_columns(model_df: DataFrame, new_columns) -> DataFrame:
    """
    create new columns for use with spans
    """
    for new_col in new_columns:
        model_df[new_col] = (None, )
    return model_df


def _drop_old_data_for_span_execution(model_df: DataFrame, removed_columns) -> DataFrame:
    # drop columns which are made superfluous by the spans to save data transfer
    model_df = model_df.drop(columns=removed_columns)
    return model_df


class SpanColumnsTokenClassificationTransformation(Transformation):
    def __init__(
            self,
            expected_input_columns: list[str],
            promised_output_columns: list[str],
            new_columns: list[str],
            removed_columns: list[str],):
        new_columns = ["entity_doc_id", "entity_char_begin", "entity_char_end"]
        removed_columns = ["text_data", "start_pos", "end_pos"]#todo as input
        super().__init__(expected_input_columns,
                         promised_output_columns,
                         new_columns,
                         removed_columns)


    def rename_columns(self, model_df: DataFrame) -> DataFrame:

        # we use different names in udf with span and without, so need to rename
        # this decision was made as to improve the naming of the columns without
        # breaking the interface of the existing udf
        model_df = model_df.rename(
            columns={"word": "entity_covered_text", "entity": "entity_type"}
        )
        return model_df

    def make_entity_span(self, df_row):
        token_doc_id = df_row["text_data_doc_id"]
        token_char_begin = df_row["start_pos"] + df_row["text_data_char_begin"]
        token_char_end = df_row["end_pos"] + df_row["text_data_char_begin"]
        return Series([token_doc_id, token_char_begin, token_char_end])

    def fill_span_columns(self, batch_df:DataFrame) -> DataFrame:
        batch_df[self.new_columns] = (
            batch_df.apply(self.make_entity_span, axis=1)
        )
        return batch_df

    def transform(self, batch_df:DataFrame) -> DataFrame:
        batch_df = _create_new_span_columns(batch_df, self.new_columns)
        batch_df = self.rename_columns(batch_df)
        batch_df = self.fill_span_columns(batch_df)
        batch_df = _drop_old_data_for_span_execution(batch_df, self.removed_columns)
        return batch_df

    def check_input_format(self, batch_df:DataFrame):
        """
        checks if all needed columns for
        transform are present, throws error otherwise
        """
        #todo
        pass

    def ensure_output_format(self, batch_df:DataFrame) -> DataFrame:
        """
        ensure all promised output columns are present
        """
        #todo
        pass




class SpanColumnsZeroShotTransformation(Transformation):
    def __init__(
            self,
            expected_input_columns: list[str],
            promised_output_columns: list[str],
            new_columns: list[str],
            removed_columns: list[str],):
        # no new span so no new columns. we just return the input span
        new_columns = []#todo as input
        removed_columns = ["text_data", "candidate_labels"]
        super().__init__(expected_input_columns,
                         promised_output_columns,
                         new_columns,
                         removed_columns)

    def transform(self, batch_df:DataFrame) -> DataFrame:
        batch_df = _create_new_span_columns(batch_df, self.new_columns)
        batch_df = _drop_old_data_for_span_execution(batch_df, self.removed_columns)
        return batch_df

    def check_input_format(self, batch_df:DataFrame):
        """
        checks if all needed columns for
        transform are present, throws error otherwise
        passes since this transform does not need any specific input columns
        """
        pass

    def ensure_output_format(self, batch_df:DataFrame) -> DataFrame:
        """
        ensure all promised output columns are present
        """
        #todo
        pass
