class MockContext:
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

    def get_dataframe(self, num_rows="all", start_col=0):
        return_df = (
            None
            if self._is_accessed_once
            else self.input_df[self.input_df.columns[start_col:]]
        )
        self._is_accessed_once = True
        return return_df
