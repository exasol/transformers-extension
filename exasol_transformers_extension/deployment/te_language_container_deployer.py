from typing import Optional
from pathlib import Path
from exasol_transformers_extension.deployment.language_container_deployer import LanguageContainerDeployer


class TeLanguageContainerDeployer(LanguageContainerDeployer):

    SLC_NAME = "exasol_transformers_extension_container_release.tar.gz"
    GH_RELEASE_URL = "https://github.com/exasol/transformers-extension/releases/download"

    def download_from_git_and_run(self, version: str,
                                  alter_system: bool = True,
                                  allow_override: bool = False) -> None:

        url = "/".join((self.GH_RELEASE_URL, version, self.SLC_NAME))
        self.download_and_run(url, self.SLC_NAME, alter_system=alter_system, allow_override=allow_override)

    def run(self, container_file: Optional[Path] = None,
            bucket_file_path: Optional[str] = None,
            alter_system: bool = True,
            allow_override: bool = False) -> None:

        if not bucket_file_path:
            bucket_file_path = self.SLC_NAME
        super().run(container_file, bucket_file_path, alter_system, allow_override)
