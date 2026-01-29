from __future__ import annotations

from pathlib import Path

from exasol_transformers_extension.deployment.default_udf_parameters import model_spec_factory, DEFAULT_BUCKETFS_CONN, \
    DEFAULT_SUBDIR
from exasol_transformers_extension.udfs.models.install_default_models_udf import InstallDefaultModelsUDF
from test.integration_tests.without_db.udfs.utils.mock_exa_environment import (
    MockExaEnvironment,
)
from test.utils.bucketfs_file_list import get_bucketfs_file_list
from test.utils.mock_connections import (
    create_mounted_bucketfs_connection,
)


from exasol.python_extension_common.connections.bucketfs_location import (
    create_bucketfs_location_from_conn_object,
)


TEST_DEFAULT_MODELS = {
    "model_for_a_specific_udf": model_spec_factory.create(model_name="prajjwal1/bert-tiny",
                                                          task_type="task",
                                                          bucketfs_conn_name=DEFAULT_BUCKETFS_CONN,
                                                          sub_dir=Path(DEFAULT_SUBDIR)),
    "model_for_another_udf": model_spec_factory.create(model_name="prajjwal1/bert-tiny",
                                                          task_type="different_task",
                                                          bucketfs_conn_name=DEFAULT_BUCKETFS_CONN,
                                                          sub_dir=Path(DEFAULT_SUBDIR)),

}

class Context:
    def __init__(self, ctx_data: list[dict[str, str]]):
        self.ctx_data = ctx_data
        self.index = 0
        self._emitted = []

    def next(self):
        self.index += 1
        return None if len(self.ctx_data) == self.index else self.index

    def emit(self, *args):
        self._emitted.append(args)

    def get_emitted(self):
        return self._emitted


class TestEnvironmentSetup:
    __test__ = False

    def __init__(self, tmp_dir: Path) :
        self.bucketfs_conn_name = DEFAULT_BUCKETFS_CONN
        self.bucketfs_connection = create_mounted_bucketfs_connection(
            tmp_dir, f"DEFAULT_BUCKETFS_CONN/"
        )

        self.ctx_data = {}

    def list_files_in_bucketfs(self):
        bucketfs_location = create_bucketfs_location_from_conn_object(
            self.bucketfs_connection
        )
        return get_bucketfs_file_list(bucketfs_location)


def test_install_default_models_udf_implementation(tmp_path):
    env1 = TestEnvironmentSetup(tmp_path)

    ctx = Context([env1.ctx_data])
    exa = MockExaEnvironment(
        {
            env1.bucketfs_conn_name: env1.bucketfs_connection,
        }
    )

    # run udf implementation
    default_models_installer = InstallDefaultModelsUDF(exa, default_model_specs=TEST_DEFAULT_MODELS)
    default_models_installer.run(ctx)

    # assertions
    env1_bucketfs_files = env1.list_files_in_bucketfs()

    expected_model_path_1 = TEST_DEFAULT_MODELS["model_for_a_specific_udf"].get_bucketfs_model_save_path()
    expected_model_path_2 = TEST_DEFAULT_MODELS["model_for_another_udf"].get_bucketfs_model_save_path()

    assert ctx.get_emitted()[0] == (
        str(expected_model_path_1),
        str(expected_model_path_1.with_suffix(".tar.gz")),
        str(expected_model_path_2),
        str(expected_model_path_2.with_suffix(".tar.gz")),
    )
    assert ctx.get_emitted()[0][1] in env1_bucketfs_files
