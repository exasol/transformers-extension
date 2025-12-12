# Getting Started

Here you find information on prerequisites for using the Transformers Extension, as well as a setup-guide.

### Table of contents

* [Prerequisites](#prerequisites)
  * [Exasol DB](#exasol-db)
  * [BucketFS Connection](#bucketfs-connection)
    * [Parameters of the Address Part of the Connection Object](#parameters-of-the-address-part-of-the-connection-object)
    * [Custom Certificates and Certificate Authorities](#custom-certificates-and-certificate-authorities)
      * [Uploading to the BucketFS of an On-Prem Database](#uploading-to-the-bucketfs-of-an-on-prem-database)
      * [Uploading to the BucketFS of a SaaS Database](#uploading-to-the-bucketfs-of-a-saas-database)
  * [Hugging Face Token](#hugging-face-token)
* [Setup](#setup)
  * [Install the Python Package](#install-the-python-package)
    * [Pip](#pip)
    * [Download and Install the Python Wheel Package](#download-and-install-the-python-wheel-package)
    * [Build the project yourself](#build-the-project-yourself)
  * [Deploy the Extension to the Database](#deploy-the-extension-to-the-database)
    * [The Pre-built Language Container](#the-pre-built-language-container)
    * [List of Options](#list-of-options)
    
## Prerequisites

### Exasol DB

* The Exasol cluster must already be running with version 7.1 or later.
* The database connection information and credentials are needed.

### BucketFS Connection

An Exasol connection object must be created with the Exasol BucketFS connection information and credentials.

Normally, the connection object is created as part of the Transformers Extension deployment (see the [Setup section](#deploy-the-extension-to-the-database) below).

This section describes how this object can be created manually.

The format of the connection object is as following:
  ```sql
  CREATE OR REPLACE CONNECTION <BUCKETFS_CONNECTION_NAME>
      TO '<BUCKETFS_ADDRESS>'
      USER '<BUCKETFS_USER>'
      IDENTIFIED BY '<BUCKETFS_PASSWORD>'
  ```

`<BUCKETFS_ADDRESS>`, `<BUCKETFS_USER>` and `<BUCKETFS_PASSWORD>` are JSON strings whose content depends on the storage backend.

Below is the description of the parameters that need to be passed for On-Prem and SaaS databases.

The distribution of the parameters among those three JSON strings do not matter.

However, we recommend to put secrets like passwords and or access tokens into the `<BUCKETFS_PASSWORD>` part.

#### Parameters of the Address Part of the Connection Object

`<BUCKET_ADDRESS>` contains various subparameters, e.g.

```sql
{"url": "https://my_cluster_11:6583", "bucket_name": "default", "service_name": "bfsdefault"}
```

The number of subparameters and their names depends on the type of database you are connecting to:

**SaaS Database**:

* `url`: Optional URL of the Exasol SaaS. Defaults to https://cloud.exasol.com.
* `account_id`: SaaS user account ID.
* `database_id`: Database ID.
* `pat`: Personal access token.

**On-Prem Database**:

* `url`: URL of the BucketFS service, e.g. `http(s)://127.0.0.1:2580`.
* `username`: BucketFS username (generally, different from the DB username, e.g. `w` for writing).
* `password`: BucketFS user password.
* `bucket_name`: Name of the bucket in the BucketFS, e.g. `default`.
* `service_name`: Name of the BucketFS service, e.g. `bfsdefault`.
* `verify`: Optional parameter that can be either a boolean, in which case it controls whether the server's TLS certificate is verified, or a string, in which case it must be a path to a CA bundle to use. Defaults to ``true``.

See section [Custom Certificates and Certificate Authorities](#custom-certificates-and-certificate-authorities) explaining the terms TLS and CA.

Here is an example of a connection object for an On-Prem database.

```sql
CREATE OR REPLACE CONNECTION "MyBucketFSConnection"
    TO '{"url":"https://my_cluster_11:6583", "bucket_name":"default", "service_name":"bfsdefault"}'
    USER '{"username":"wxxxy"}'
    IDENTIFIED BY '{"password":"wrx1t09x9e"}';
```

For more information, please check the [Create Connection in Exasol](https://docs.exasol.com/sql/create_connection.htm) document.

#### Custom Certificates and Certificate Authorities

Certificates are required for secure network communication using Transport Layer Security (TLS).

Each certificate is issued by a Certificate Agency (CA) guaranteeing its validy and security in a chain of trusted certificates, see also the [TLS Tutorial](https://github.com/exasol/exasol-java-tutorial/blob/main/tls-tutorial/doc/tls_introduction.md#certificates-and-certification-agencies).

For using a custom CA bundle, you first need to upload it to the BucketFS.

##### Uploading to the BucketFS of an On-Prem Database

The following command puts a bundle in a single file called `ca_bundle.pem` to the bucket `bucket1` in the subdirectory `tls`:

```Shell
curl -T ca_bundle.pem https://w:w-password@192.168.6.75:1234/bucket1/tls/ca_bundle.pem
```

For more details on uploading files to the BucketFS, see the [Exasol documentation](https://docs.exasol.com/db/latest/database_concepts/bucketfs/file_access.htm).

##### Uploading to the BucketFS of a SaaS Database

Please use the [Exasol SaaS REST API](https://cloud.exasol.com/openapi/index.html#/Files) for uploading files to the BucketFS on a SaaS database. The CA bundle path should have the following format:

```
/buckets/<service-name>/<bucket-name>/<path-to-the-file-or-directory>
```

For example, if the service name is ``bfs_service1`` and the bundle was uploaded with the above curl command, the path should look like ``/buckets/bfs_service1/bucket1/tls/ca_bundle.pem``. Please note that for the BucketFS on a SaaS database, the service and bucket names are fixed at respectively ``upload`` and ``default``.

### Hugging Face Token

A valid token is required to download private models from the Hugging Face hub and later generate predictions with them.

This token is considered sensitive information; hence, it should be stored in an Exasol Connection object.

The easiest way to do this is to provide the token as an option during the extension deployment, see section [Setup section](#deploy-the-extension-to-the-database) below.

You can also create a connection manually:

```sql
CREATE OR REPLACE CONNECTION <TOKEN_CONNECTION_NAME>
    TO ''
    IDENTIFIED BY '<PRIVATE_MODEL_TOKEN>'
```

For more information, please check the [Create Connection in Exasol](https://docs.exasol.com/sql/create_connection.htm) document.


## Setup

### Install the Python Package

There are multiple ways to install the Python package:
* [Pip install](#pip)
* [Download the wheel from GitHub](#download-and-install-the-python-wheel-package)
* [Build the project yourself](#build-the-project-yourself)

Additionally, you will need a Script Language Container. For instructions, see [The Pre-built Language Container](#the-pre-built-language-container) section.

#### Pip

The Transformers Extension is published on [Pypi](https://pypi.org/project/exasol-transformers-extension/).

You can install it with:

```shell
pip install exasol-transformers-extension
```

#### Download and Install the Python Wheel Package

You can also get the wheel from a GitHub release.

* The latest version of the Python package of this extension can be downloaded from the [GitHub Release](https://github.com/exasol/transformers-extension/releases/latest). Please download the following built archive:
  ```buildoutcfg
  exasol_transformers_extension-<version-number>-py3-none-any.whl
  ```

If you need to use a version < 0.5.0, the build archive is called `transformers_extension.whl`.

Then, install the packaged Transformers Extension project as follows:

```shell
pip install <path/wheel-filename.whl>
```

#### Build the project yourself

To build Transformers Extension yourself, you need to have [Poetry](https://python-poetry.org/) (>= 2.1.0) installed.

Then, you will need to clone the GitHub repository https://github.com/exasol/transformers-extension/ and install and build the project as follows:

```bash
poetry install
poetry build
```

### Deploy the Extension to the Database

The Transformers Extension must be deployed to the database using the following command:

```shell
python -m exasol_transformers_extension.deploy <options>
```

#### The Pre-built Language Container

The deployment includes the installation of the Script Language Container (SLC). The SLC is a way to install the required programming language and necessary dependencies in the Exasol database so that UDF scripts can be executed. The version of the installed SLC must match the version of the Transformers Extension package.  See [the latest release](https://github.com/exasol/transformers-extension/releases) on GitHub.

#### List of Options

For information about the available options common to all Exasol extensions, please refer to the [documentation](https://github.com/exasol/python-extension-common/blob/main/doc/user_guide/user-guide.md#language-container-deployer) in the Exasol Python Extension Common package.

In addition, this extension provides the following installation options:

| Option name             | Default   | Comment                                                                 |
|-------------------------|-----------|-------------------------------------------------------------------------|
| `--[no-]deploy-slc`     | True      | Install SLC as part of the deployment                                   |
| `--[no-]deploy-scripts` | True      | Install scripts as part of the deployment                               |
| `--bucketfs-conn-name`  |           | Name of the [BucketFS connection object](#bucketfs-connection)          |
| `--token-conn-name`     |           | Name of the [token connection object](#huggingface-token) if required   |
| `--token`               |           | The [Huggingface token](#huggingface-token) if required                 |

The connection objects will not be created if their names are not provided.
