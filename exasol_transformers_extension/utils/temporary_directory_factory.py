import tempfile


class TemporaryDirectoryFactory:

    def create(self) -> tempfile.TemporaryDirectory:
        return tempfile.TemporaryDirectory()
