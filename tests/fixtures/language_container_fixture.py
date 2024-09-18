import pytest

from exasol_transformers_extension.deployment.language_container import language_container_factory


@pytest.fixture(scope='session')
def slc_builder(use_onprem, use_saas):
    if use_onprem or use_saas:
        with language_container_factory() as container_builder:
            yield container_builder
    else:
        yield None
