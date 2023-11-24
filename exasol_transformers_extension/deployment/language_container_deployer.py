from enum import Enum
import pyexasol
from typing import List, Optional
from pathlib import Path, PurePosixPath
from exasol_bucketfs_utils_python.bucketfs_location import BucketFSLocation
import logging
from exasol_transformers_extension.utils.bucketfs_operations import \
    create_bucketfs_location
from exasol_transformers_extension.deployment.deployment_utils import get_websocket_ssl_options

logger = logging.getLogger(__name__)


class LanguageActiveLevel(Enum):
    f"""
    Language activation level, i.e.
    ALTER <LanguageActiveLevel> SET SCRIPT_LANGUAGES=...
    """
    Session = 'SESSION'
    System = 'SYSTEM'


class LanguageContainerDeployer:
    def __init__(self,
                 pyexasol_connection: pyexasol.ExaConnection,
                 language_alias: str,
                 bucketfs_location: BucketFSLocation,
                 container_file: Path):
        self._container_file = container_file
        self._bucketfs_location = bucketfs_location
        self._language_alias = language_alias
        self._pyexasol_conn = pyexasol_connection
        logger.debug(f"Init {LanguageContainerDeployer.__name__}")

    def deploy_container(self, allow_override: bool = False) -> None:
        """
        Uploads the SLC and activates it at the SYSTEM level.

        allow_override - If True the activation of a language container with the same alias will be overriden,
                         otherwise a RuntimeException will be thrown.
        """
        path_in_udf = self.upload_container()
        self.activate_container(LanguageActiveLevel.System, allow_override,  path_in_udf)

    def upload_container(self) -> PurePosixPath:
        """
        Uploads the SLC.
        Returns the path where the container is uploaded as it's seen by a UDF.
        """
        if not self._container_file.is_file():
            raise RuntimeError(f"Container file {self._container_file} "
                               f"is not a file.")
        with open(self._container_file, "br") as f:
            upload_uri, path_in_udf = \
                self._bucketfs_location.upload_fileobj_to_bucketfs(
                    fileobj=f, bucket_file_path=self._container_file.name)
        logging.debug("Container is uploaded to bucketfs")
        return PurePosixPath(path_in_udf)

    def activate_container(self, alter_type: LanguageActiveLevel = LanguageActiveLevel.Session,
                           allow_override: bool = False,
                           path_in_udf: Optional[PurePosixPath] = None) -> None:
        """
        Activates the SLC container at the required level.

        alter_type     - Language activation level, defaults to the SESSION.
        allow_override - If True the activation of a language container with the same alias will be overriden,
                         otherwise a RuntimeException will be thrown.
        path_in_udf    - If known, a path where the container is uploaded as it's seen by a UDF.
        """
        alter_command = self.generate_activation_command(alter_type, allow_override, path_in_udf)
        self._pyexasol_conn.execute(alter_command)
        logging.debug(alter_command)

    def generate_activation_command(self, alter_type: LanguageActiveLevel,
                                    allow_override: bool = False,
                                    path_in_udf: Optional[PurePosixPath] = None) -> str:
        """
        Generates an SQL command to activate the SLC container at the required level. The command will
        preserve existing activations of other containers identified by different language aliases.
        Activation of a container with the same alias, if exists, will be overwritten.

        alter_type     - Activation level - SYSTEM or SESSION.
        allow_override - If True the activation of a language container with the same alias will be overriden,
                         otherwise a RuntimeException will be thrown.
        path_in_udf    - If known, a path where the container is uploaded as it's seen by a UDF.
        """
        if path_in_udf is None:
            path_in_udf = self._bucketfs_location.generate_bucket_udf_path(self._container_file.name)
        new_settings = \
            self._update_previous_language_settings(alter_type, allow_override, path_in_udf)
        alter_command = \
            f"ALTER {alter_type.value} SET SCRIPT_LANGUAGES='{new_settings}';"
        return alter_command

    def _update_previous_language_settings(self, alter_type: LanguageActiveLevel,
                                           allow_override: bool,
                                           path_in_udf: PurePosixPath) -> str:
        prev_lang_settings = self._get_previous_language_settings(alter_type)
        prev_lang_aliases = prev_lang_settings.split(" ")
        self._check_if_requested_language_alias_already_exists(
            allow_override, prev_lang_aliases)
        new_definitions_str = self._generate_new_language_settings(
            path_in_udf, prev_lang_aliases)
        return new_definitions_str

    def _generate_new_language_settings(self, path_in_udf: PurePosixPath,
                                        prev_lang_aliases: List[str]) -> str:
        other_definitions = [
            alias_definition for alias_definition in prev_lang_aliases
            if not alias_definition.startswith(self._language_alias + "=")]
        path_in_udf_without_bucksts = Path(*path_in_udf.parts[2:])
        new_language_alias_definition = \
            f"{self._language_alias}=localzmq+protobuf:///" \
            f"{path_in_udf_without_bucksts}?lang=python#" \
            f"{path_in_udf}/exaudf/exaudfclient_py3"
        new_definitions = other_definitions + [new_language_alias_definition]
        new_definitions_str = " ".join(new_definitions)
        return new_definitions_str

    def _check_if_requested_language_alias_already_exists(
            self, allow_override: bool,
            prev_lang_aliases: List[str]) -> None:
        definition_for_requested_alias = [
            alias_definition for alias_definition in prev_lang_aliases
            if alias_definition.startswith(self._language_alias + "=")]
        if not len(definition_for_requested_alias) == 0:
            warning_message = f"The requested language alias {self._language_alias} is already in use."
            if allow_override:
                logging.warning(warning_message)
            else:
                raise RuntimeError(warning_message)

    def _get_previous_language_settings(self, alter_type: LanguageActiveLevel) -> str:
        result = self._pyexasol_conn.execute(
            f"""SELECT "{alter_type.value}_VALUE" FROM SYS.EXA_PARAMETERS WHERE 
            PARAMETER_NAME='SCRIPT_LANGUAGES'""").fetchall()
        return result[0][0]

    @classmethod
    def create(cls, bucketfs_name: str, bucketfs_host: str, bucketfs_port: int,
            bucketfs_use_https: bool, bucketfs_user: str, container_file: Path,
            bucketfs_password: str, bucket: str, path_in_bucket: str,
            dsn: str, db_user: str, db_password: str, language_alias: str,
            ssl_cert_path: str = None, use_ssl_cert_validation: bool = True) -> "LanguageContainerDeployer":

        websocket_sslopt = get_websocket_ssl_options(use_ssl_cert_validation, ssl_cert_path)

        pyexasol_conn = pyexasol.connect(
            dsn=dsn,
            user=db_user,
            password=db_password,
            encryption=True,
            websocket_sslopt=websocket_sslopt
        )

        bucketfs_location = create_bucketfs_location(
            bucketfs_name, bucketfs_host, bucketfs_port, bucketfs_use_https,
            bucketfs_user, bucketfs_password, bucket, path_in_bucket)

        return cls(pyexasol_conn, language_alias, bucketfs_location, container_file)
