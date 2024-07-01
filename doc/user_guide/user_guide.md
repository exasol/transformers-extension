
# User Guide

The Transformers Extension provides a Python library with UDFs that allow the 
use of pre-trained NLP models provided by the [Transformers API](https://huggingface.co/docs/transformers/index).

The extension provides two types of UDFs:
 - DownloaderUDF :  It is responsible to download the specified pre-defined model into the Exasol BucketFS.
 - Prediction UDFs: These are a group of UDFs for each supported task. Each of them uses the downloaded pre-trained
model and perform prediction. These are the supported tasks:
   1. Sequence Classification for Single Text 
   2. Sequence Classification for Text Pair
   3. Question Answering
   4. Masked Language Modelling
   5. Text Generation
   6. Token Classification
   7. Text Translation
   8. Zero-Shot Text Classification


## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Setup](#setup)
- [Model Downloader UDF](#model-downloader-udf)
- [Prediction UDFs](#prediction-udfs)
  1. [Sequence Classification for Single Text UDF](#sequence-classification-for-single-text-udf)
  2. [Sequence Classification for Text Pair UDF](#sequence-classification-for-text-pair-udf)
  3. [Question Answering UDF](#question-answering-udf)
  4. [Masked Language Modelling UDF](#masked-language-modelling-udf)
  5. [Text Generation UDF](#text-generation-udf)
  6. [Token Classification UDF](#token-classification-udf)
  7. [Text Translation UDF](#text-translation-udf)
  8. [Zero-Shot Text Classification](#zero-shot-text-classification-udf)

## Introduction

This Exasol Extension provides UDFs for interacting with Hugging Face's Transformers API in order to use 
pre-trained models on an Exasol Cluster.

User Defined Function, UDFs for short, are scripts in various programming languages that can be 
executed in the Exasol Database. They can be used by the user for more flexibility in data processing. 
In this Extension we provide multiple UDFs for you to use on your Exasol Database.
You can find a more detailed documentation on UDFs
[here](https://docs.exasol.com/db/latest/database_concepts/udf_scripts.htm).

UDFs and the necessary [Script language Container](https://docs.exasol.com/db/latest/database_concepts/udf_scripts/adding_new_packages_script_languages.htm) 
are stored in Exasol's file system BucketFS, and we also use this to store the Hugging Face
models on the Exasol Cluster. More information on The BucketFS can be found 
[here](https://docs.exasol.com/db/latest/database_concepts/bucketfs/bucketfs.htm).

## Getting Started

### Exasol DB
- The Exasol cluster must already be running with version 7.1 or later.
- DB connection information and credentials are needed.

### BucketFS Connection 
An Exasol connection object must be created with Exasol BucketFS connection information and credentials. 
The format of the connection object is as following: 
  ```sql
  CREATE OR REPLACE CONNECTION <BUCKETFS_CONNECTION_NAME>
      TO '<BUCKETFS_ADDRESS>'
      USER '<BUCKETFS_USER>'
      IDENTIFIED BY '<BUCKETFS_PASSWORD>'
  ```
`<BUCKETFS_ADDRESS>`, `<BUCKETFS_USER>` and `<BUCKETFS_PASSWORD>` are JSON strings whose content depends on the storage backend.
Below is the description of the parameters that need to be passed for On-Prem and SaaS databases. The distribution of
the parameters among those three JSON strings do not matter. However, we recommend to put secrets like passwords and or access tokens
into the `<BUCKETFS_PASSWORD>` part.

**On-Prem Database**
- url: Url of the BucketFS service, e.g. "http(s)://127.0.0.1:2580".
- username: BucketFS username (generally, different from the DB username).
- password: BucketFS user password.
- bucket_name: Name of the bucket in the BucketFS.
- verify: Optional parameter that can be either a boolean, in which case it controls whether the server's
    TLS certificate is verified, or a string, in which case it must be a path to a CA bundle to use. Defaults to ``true``.
    To use a custom CA bundle, firstly it needs to be uploaded to the BucketFS. Below is an example curl command
    that puts a bundle in a single file called `ca_bundle.pem` to the bucket `bucket1` in a subdirectory `tls`:
    ```commandline
        curl -T ca_bundle.pem https://w:w-password@192.168.6.75:1234/bucket1/tls/ca_bundle.pem
    ```
    For more details on uploading files to the BucketFS see the [Exasol documentation](https://docs.exasol.com/db/latest/database_concepts/bucketfs/file_access.htm).
    Please use the [Exasol SaaS REST API](https://cloud.exasol.com/openapi/index.html#/Files) for uploading files to the BucketFS on a SaaS database.
    The CA bundle path should have the following format:
    ```
        /buckets/<service-name>/<bucket-name>/<path-to-the-file-or-directory>
    ```
    For example, if the service name is ``bfs_service1`` and the bundle was uploaded with the above curl command, the path should look like
    ``/buckets/bfs_service1/bucket1/tls/ca_bundle.pem``.
    Please note that for the BucketFS on a SaaS database, the service and bucket names are fixed at
    respectively ``upload`` and ``default``.
- service_name: Name of the BucketFS service.

**SaaS Database**
- url: Optional rrl of the Exasol SaaS. Defaults to 'https://cloud.exasol.com'.
- account_id: SaaS user account ID.
- database_id: Database ID. 
- pat: Personal Access Token.

Here is an example of a connection object for an On-Prem database.
  ```sql
  CREATE OR REPLACE CONNECTION "MyBucketFSConnection"
      TO '{"url":"https://my_cluster_11:6583", "bucket_name":"default", "service_name":"bfsdefault"}'
      USER '{"username":"wxxxy"}'
      IDENTIFIED BY '{"password":"wrx1t09x9e"}';
  ```
For more information please check the [Create Connection in Exasol](https://docs.exasol.com/sql/create_connection.htm?Highlight=connection) document.

### Huggingface token
A valid token is required to download private models from the Huggingface hub and run prediction on them. 
To avoid exposing such sensitive information, you can use Exasol Connection 
objects. As seen in the example below, a token can be specified in the 
password part of the Exasol connection object:
  ```sql
  CREATE OR REPLACE CONNECTION <TOKEN_CONNECTION_NAME>
      TO ''
      IDENTIFIED BY '<PRIVATE_MODEL_TOKEN>'
  ```
  
## Setup
### Install the Python Package

There are multiple ways to install the Python Package. You can use Pip install, 
Download the Wheel from GitHub or build the project yourself.
Additionally, you will need a Script Language Container. Find the how-to below.

#### Pip

The Transformers Extension is published on [Pypi](https://pypi.org/project/exasol-transformers-extension/). 

You can install it with:

```shell
pip install exasol-transformers-extension
```


#### Download and Install the Python Wheel Package

You can also get the wheel from a GitHub release.
- The latest version of the Python package of this extension can be 
downloaded from the [GitHub Release](https://github.com/exasol/transformers-extension/releases/latest).
Please download the following built archive:
```buildoutcfg 
exasol_transformers_extension-<version-number>-py3-none-any.whl
```
If you need to use a version < 0.5.0, the build archive is called `transformers_extension.whl`.

Then install the packaged transformers-extension project as follows:
```shell
pip install <path/wheel-filename.whl>
```

#### Build the project yourself

In order to build Transformers Extension yourself, you need to have the [Poetry](https://python-poetry.org/)
(>= 1.1.11) package manager installed. Clone the GitHub Repository, and install and build 
the `transformers-extension` as follows:
```bash
poetry install
poetry build
```

### The Pre-built Language Container

This extension requires the installation of a Language Container in the Exasol Database. 
The Script Language Container is a way to install the required programming language and 
necessary dependencies in the Exasol Database so the UDF scripts can be executed.

The Language Container is downloaded and installed by executing the 
deployment script below. Please make sure that the version of the Language Container matches the
installed version of the Transformers Extension Package. See [the latest release](https://github.com/exasol/transformers-extension/releases) on GitHub.

  ```buildoutcfg
  python -m exasol_transformers_extension.deploy language-container <options>
  ```

Please refer to the [Language Container Deployment Guide] for details about this command. 

### Scripts Deployment

Next you need to deploy all necessary scripts to the specified `SCHEMA` in your Exasol DB with the same
`LANGUAGE_ALIAS` using the following Python CLI command:

  ```buildoutcfg
  python -m exasol_transformers_extension.deploy scripts <options>
  ```

The choice of options is primarily determined by the storage backend being used - On-Prem or SaaS.

### List of options

The table below lists all available options. It shows which ones are applicable for On-Prem and for SaaS backends.
Unless stated otherwise in the comments column, the option is required for either or both backends.

Some of the values, like passwords, are considered confidential. For security reasons, it is recommended to store
those values in environment variables instead of providing them in the command line. The names of the environment
variables are given in the comments column, where applicable. Alternatively, it is possible to put just the name of
an option in the command line, without providing its value. In this case, the command will prompt to enter the value
interactively. For long values, such as the SaaS account id, it is more practical to copy/paste the value from
another source.

| Option name                  | On-Prem | SaaS | Comment                                         |
|:-----------------------------|:-------:|:----:|:------------------------------------------------|
| dsn                          |   [x]   |      | i.e. <db_host:db_port>                          |
| db-user                      |   [x]   |      |                                                 |
| db-pass                      |   [x]   |      | Env. [DB_PASSWORD]                              |
| saas-url                     |         | [x]  | Optional, Env. [SAAS_HOST]                      |
| saas-account-id              |         | [x]  | Env. [SAAS_ACCOUNT_ID]                          |
| saas-database-id             |         | [x]  | Optional, Env. [SAAS_DATABASE_ID]               |
| saas-database-name           |         | [x]  | Optional, provide if the database_id is unknown |
| saas-token                   |         | [x]  | Env. [SAAS_TOKEN]                               |
| language-alias               |   [x]   | [x]  | Optional                                        |
| schema                       |   [x]   | [x]  | DB schema to deploy the scripts in              |
| ssl-cert-path                |   [x]   | [x]  | Optional                                        |
| [no_]use-ssl-cert-validation |   [x]   | [x]  | Optional boolean, defaults to True              |
| ssl-client-cert-path         |   [x]   |      | Optional                                        |
| ssl-client-private-key       |   [x]   |      | Optional                                        |

### TLS/SSL options

The `--ssl-cert-path` is needed if the TLS/SSL certificate is not in the OS truststore.
Generally speaking, this certificate is a list of trusted CA. It is needed for the server's certificate
validation by the client.
The option `--use-ssl-cert-validation`is the default, it can be disabled with `--no-use-ssl-cert-validation`.
One needs to exercise caution when turning the certificate validation off as it potentially lowers the security of the
Database connection.
The "server" certificate described above shall not be confused with the client's own certificate.
In some cases, this certificate may be requested by a server. The client certificate may or may not include
the private key. In the latter case, the key may be provided as a separate file.

## Store Models in BucketFS
Before you can use pre-trained models, the models must be stored in the 
BucketFS. We provide two different ways to load transformers models 
into the BucketFS. You may either use the Model Downloader UDF to download a Hugging Face 
transformers model directly from the Exasol Database, or you can download the model to your local 
file system and upload it to the Database using the Model Uploader Script.
The Model Downloader UDF is the simpler option, but if you do not want to connect your Exasol Database
directly to the internet, the Model Uploader Script is an option for you.

Note that the extension currently only supports the `PyTorch` framework. 
Please make sure that the selected models are in the `Pytorch` model library section.

### 1. Model Downloader UDF
Using the `TE_MODEL_DOWNLOADER_UDF` below, you can download the desired model 
from the huggingface hub and upload it to BucketFS.
This requires the Exasol Database to have internet access, since the UDF will 
download the model from Hugging Face to the Database without saving it somewhere else intermittently.
If you are using the Exasol DockerDB or an Exasol version 8 setup via 
[c4](https://docs.exasol.com/db/latest/administration/on-premise/admin_interface/c4.htm), 
this is not the case by default, and you need to specify a name server.
For example setting it to 'nameserver = 8.8.8.8' will set it to use Google DNS.
You will need to use [ConfD](https://docs.exasol.com/db/latest/confd/confd.htm) to do this, 
you can use the [general_settings](https://docs.exasol.com/db/latest/confd/jobs/general_settings.htm) command.
If you are using the [Integration Test Docker Environment](https://github.com/exasol/integration-test-docker-environment), 
you can just set the nameserver parameter like this: `--nameserver 8.8.8.8`

Once you have internet access, invoke the UDF like this:

```sql
SELECT TE_MODEL_DOWNLOADER_UDF(
    model_name,
    sub_dir,
    bucketfs_conn,
    token_conn
)
```
- Parameters:
  - ```model_name```: The name of the model to use for prediction. You can find the 
  details of the models on the [huggingface models page](https://huggingface.co/models).
  - ```sub_dir```: The directory where the model is stored in the BucketFS.
  - ```bucketfs_conn```: The BucketFS connection name.
  - ```token_conn```: The connection name containing the token required for 
  private models. You can use an empty string ('') for public models. For details 
  on how to create a connection object with token information, please check 
  [here](#getting-started).
  
    
### 2. Model Uploader Script
You can invoke the python script as below which allows to load the transformer 
models from the local filesystem into BucketFS:

  ```buildoutcfg
  python -m exasol_transformers_extension.upload_model <options>
  ```

#### List of options

The table below lists all available options. It shows which ones are applicable for On-Prem and for SaaS backends.
Unless stated otherwise in the comments column, the option is required for either or both backends.

| Option name                  | On-Prem | SaaS | Comment                                         |
|:-----------------------------|:-------:|:----:|:------------------------------------------------|
| bucketfs-name                |   [x]   |      |                                                 |
| bucketfs-host                |   [x]   |      |                                                 |
| bucketfs-port                |   [x]   |      |                                                 |
| bucketfs-user                |   [x]   |      |                                                 |
| bucketfs-password            |   [x]   |      | Env. [BUCKETFS_PASSWORD]                        |
| bucketfs-use-https           |   [x]   |      | Optional boolean, defaults to True              |
| bucket                       |   [x]   |      |                                                 |
| saas-url                     |         | [x]  |                                                 |
| saas-account-id              |         | [x]  | Env. [SAAS_ACCOUNT_ID]                          |
| saas-database-id             |         | [x]  | Optional, Env. [SAAS_DATABASE_ID]               |
| saas-database-name           |         | [x]  | Optional, provide if the database_id is unknown |
| saas-token                   |         | [x]  | Env. [SAAS_TOKEN]                               |
| model-name                   |   [x]   | [x]  |                                                 |
| path-in-bucket               |   [x]   | [x]  | Root location in the bucket for all models      |
| sub-dir                      |   [x]   | [x]  | Sub-directory where this model should be stored |
| [no_]use-ssl-cert-validation |   [x]   | [x]  | Optional boolean, defaults to True              |

**Note**: The options --local-model-path needs to point to a path which contains the model and its tokenizer. 
These should have been saved using transformers [save_pretrained](https://huggingface.co/docs/transformers/v4.32.1/en/installation#fetch-models-and-tokenizers-to-use-offline) 
function to ensure proper loading by the Transformers Extension UDFs.
You can download the model using python like this:

```python
    for model_factory in [transformers.AutoModel, transformers.AutoTokenizer]:
        # download the model and tokenizer from Hugging Face
        model = model_factory.from_pretrained(model_name)
        # save the downloaded model using the save_pretrained function
        model_save_path = <your local model save path>
        model.save_pretrained(model_save_path)
```
***Note:*** Hugging Face models consist of two parts, the model and the tokenizer. 
Make sure to download and save both into the same save directory so the upload model script uploads them together.
And then upload it using exasol_transformers_extension.upload_model script where ```--local-model-path = <your local model save path>```


## Using Prediction UDFs
We provide 7 prediction UDFs in this Transformers Extension, each performing an NLP 
task through the [transformers API](https://huggingface.co/docs/transformers/task_summary). 
These tasks use the model downloaded to BucketFS and run inference using 
the user-supplied inputs.

### Sequence Classification for Single Text UDF
This UDF classifies the given single text  according to a given number of 
classes of the specified  model. An example usage is given below:
```sql
SELECT TE_SEQUENCE_CLASSIFICATION_SINGLE_TEXT_UDF(
    device_id,
    bucketfs_conn,
    sub_dir,
    model_name,
    text_data
)
```
- Parameters:
  - ```device_id```: To run on GPU, specify the valid cuda device ID. Otherwise, 
  you can provide NULL for this parameter.
  - ```bucketfs_conn```: The BucketFS connection name
  - ```sub_dir```: The directory where the model is stored in the BucketFS.
  - ```model_name```: The name of the model to use for prediction. You can find the 
  details of the models in [huggingface models page](https://huggingface.co/models).
  - ```text_data```: The input text to be classified

The inference results are presented with predicted _LABEL_ and confidence 
 _SCORE_ columns, combined with the inputs used when calling 
this UDF. In case of any error during model loading or prediction, these new 
columns are set to `null` and column _ERROR_MESSAGE_ is set 
to the stacktrace of the error. For example:

| BUCKETFS_CONN | SUB_DIR | MODEL_NAME | TEXT_DATA | LABEL   | SCORE | ERROR_MESSAGE |
|---------------|---------|------------|-----------|---------|-------|---------------|
| conn_name     | dir/    | model_name | text      | label_1 | 0.75  | None          |
| ...           | ...     | ...        | ...       | ...     | ...   | ...           |


### Sequence Classification for Text Pair UDF
This UDF takes two input sequences and compares them, e.g., it is used to 
determine if two sequences are paraphrases of each other. An example usage is given below:
```sql
SELECT TE_SEQUENCE_CLASSIFICATION_TEXT_PAIR_UDF(
    device_id,
    bucketfs_conn,
    sub_dir,
    model_name,
    first_text,
    second_text
)
```
- Parameters:
  - ```device_id```: To run on GPU, specify the valid cuda device ID. Otherwise, 
  you can provide NULL for this parameter.
  - ```bucketfs_conn```: The BucketFS connection name
  - ```sub_dir```: The directory where the model is stored in the BucketFS.
  - ```model_name```: The name of the model to use for prediction. You can find the 
  details of the models in [huggingface models page](https://huggingface.co/models).
  - ```first_text```: The first input text
  - ```second_text```: The second input text

The inference results are presented with predicted _LABEL_ and confidence 
 _SCORE_ columns, combined with the inputs used when calling this UDF. 
In case of any error during model loading or prediction, these new 
columns are set to `null` and column _ERROR_MESSAGE_ is set 
to the stacktrace of the error. 


### Question Answering UDF
This UDF extracts answer(s) from a given question text. With the `top_k` 
input parameter, up to `k` answers with the best inference scores can be returned. 
An example usage is given below:
```sql
SELECT TE_QUESTION_ANSWERING_UDF(
    device_id,
    bucketfs_conn,
    sub_dir,
    model_name,
    question,
    context_text,
    top_k
)
```
- Parameters:
  - ```device_id```: To run on GPU, specify the valid cuda device ID. Otherwise, 
  you can provide NULL for this parameter.
  - ```bucketfs_conn```: The BucketFS connection name
  - ```sub_dir```: The directory where the model is stored in the BucketFS.
  - ```model_name```: The name of the model to use for prediction. You can find the 
  details of the models in [huggingface models page](https://huggingface.co/models).
  - ```question```: The question text
  - ```context_text```: The context text, associated with question
  - ```top_k```: The number of answers to return. Note that, `k` number of answers are not guaranteed. If there are not enough options 
in the context, it might return less than `top_k` answers (see the [top_k parameter of QuestionAnswering](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.QuestionAnsweringPipeline.__call__.topk)).

The inference results are presented with predicted _ANSWER_, confidence 
 _SCORE_, and _RANK_ columns, combined with the inputs used when calling this UDF.
If `top_k` > 1, each input row is repeated for each answer. In case of any error 
during model loading or prediction, these new columns are set to `null` and column _ERROR_MESSAGE_ is set 
to the stacktrace of the error. For example:

| BUCKETFS_CONN | SUB_DIR | MODEL_NAME | QUESTION   | CONTEXT   | TOP_K | ANSWER   | SCORE | RANK | ERROR_MESSAGE |
|---------------|---------|------------|------------|-----------|-------|----------|-------|------|---------------|
| conn_name     | dir/    | model_name | question_1 | context_1 | 2     | answer_1 | 0.75  | 1    | None          |
| conn_name     | dir/    | model_name | question_2 | context_1 | 2     | answer_2 | 0.70  | 2    | None          |
| ...           | ...     | ...        | ...        | ...       | ...   | ...      | ...   | ..   | ...           |


### Masked Language Modelling UDF
This UDF is responsible for masking tokens in a given text with a masking token, 
and then filling that masks with appropriate tokens. The masking token of 
this UDF is ```<mask>```. An example usage is given below:
```sql
SELECT TE_FILLING_MASK_UDF(
    device_id,
    bucketfs_conn,
    sub_dir,
    model_name,
    text_data,
    top_k
)
```

- Parameters:
  - ```device_id```: To run on GPU, specify the valid cuda device ID. Otherwise, 
  you can provide NULL for this parameter.
  - ```bucketfs_conn```: The BucketFS connection name
  - ```sub_dir```: The directory where the model is stored in the BucketFS.
  - ```model_name```: The name of the model to use for prediction. You can find the 
  details of the models in [huggingface models page](https://huggingface.co/models).
  - ```text_data```: The text data containing masking tokens
  - ```top_k```: The number of predictions to return.


The inference results are presented with _FILLED_TEXT_, confidence 
 _SCORE_, and _RANK_ columns, combined with the inputs used when calling this UDF.
If `top_k` > 1, each input row is repeated for each prediction. In case of any 
error during model loading or prediction, these new columns are set to `null` 
and column _ERROR_MESSAGE_ is set to the stacktrace of the error. For example:

| BUCKETFS_CONN | SUB_DIR | MODEL_NAME | TEXT_DATA     | TOP_K | FILLED_TEXT   | SCORE | RANK | ERROR_MESSAGE |
| ------------- |---------|------------|---------------| ----- |---------------| ----- |------|---------------|
| conn_name     | dir/    | model_name | text `<mask>` | 2     | text filled_1 | 0.75  |   1  | None          |
| conn_name     | dir/    | model_name | text `<mask>` | 2     | text filled_2 | 0.70  |   2  | None          |
| ...           | ...     | ...        | ...           | ...   | ...           | ...   |  ... | ...           |


### Text Generation UDF
This UDF aims to consistently predict the continuation of the given text. 
The length of the text to be generated is limited by the `max_length` parameter.
An example usage is given below:

```sql
SELECT TE_TEXT_GENERATION_UDF(
    device_id,
    bucketfs_conn,
    sub_dir,
    model_name,
    text_data,
    max_length,
    return_full_text
)
```
- Parameters:
  - ```device_id```: To run on GPU, specify the valid cuda device ID. Otherwise, 
  you can provide NULL for this parameter.
  - ```bucketfs_conn```: The BucketFS connection name.
  - ```sub_dir```: The directory where the model is stored in the BucketFS.
  - ```model_name```: The name of the model to use for prediction. You can find the 
  details of the models in [huggingface models page](https://huggingface.co/models).
  - ```text_data```: The context text.
  - ```max_length```: The maximum total length of text to be generated.
  - ```return_full_text```:  If set to False only added text is returned, otherwise the full text is returned.

The inference results are presented with _GENERATED_TEXT_ column, 
combined with the inputs used when calling this UDF. In case of any error during 
model loading or prediction, these new columns are set to `null`, and you can 
see the stacktrace of the error in the _ERROR_MESSAGE_ column.


### Token Classification UDF
The main goal of this UDF is to  assign a label to individual tokens in a  given text. 
There are two popular subtasks of token classification: 
 - Named Entity Recognition (NER) which identifies specific entities in a text, such as dates, people, and places. 
 - Part of Speech (PoS) which identifies which words in a text are verbs, nouns, and punctuation.

```sql
SELECT TE_TOKEN_CLASSIFICATION_UDF(
    device_id,
    bucketfs_conn,
    sub_dir,
    model_name,
    text_data,
    aggregation_strategy
)
```
- Parameters:
  - ```device_id```: To run on GPU, specify the valid cuda device ID. Otherwise, 
  you can provide NULL for this parameter.
  - ```bucketfs_conn```: The BucketFS connection name.
  - ```sub_dir```: The directory where the model is stored in the BucketFS.
  - ```model_name```: The name of the model to use for prediction. You can find the 
  details of the models in [huggingface models page](https://huggingface.co/models).
  - ```text_data```: The text to analyze.
  - ```aggregation_strategy```:  The strategy to fuse (or not) tokens based on the model prediction. 
  It is set to `simple` strategy by default, if you supply NULL. Please check [here](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.TokenClassificationPipeline.aggregation_strategy) 
  for more information.
 

The inference results are presented with _START_POS_ indicating the index of the starting character of the token, 
_END_POS_ indicating the index of the ending character of the token, _WORD_ indicating the token, predicted _ENTITY_, and 
confidence _SCORE_ columns, combined with the inputs used when calling this UDF.
In case of any error during model loading or prediction, these new 
columns are set to `null`, and column _ERROR_MESSAGE_ is set 
to the stacktrace of the error. For example:

| BUCKETFS_CONN | SUB_DIR | MODEL_NAME | TEXT_DATA | AGGREGATION_STRATEGY | START_POS | END_POS | WORD | ENTITY | SCORE | ERROR_MESSAGE |
| ------------- |---------|------------|-----------|----------------------|-----------|---------|------|--------|-------| ------------- |
| conn_name     | dir/    | model_name | text      | simple               | 0         | 4       | text | noun   | 0.75  | None          |
| ...           | ...     | ...        | ...       | ...                  | ...       | ...     | ...  | ..     | ...   | ...           |



### Text Translation UDF
This UDF translates a given text from one language to another.

```sql
SELECT TE_TRANSLATION_UDF(
    device_id,
    bucketfs_conn,
    sub_dir,
    model_name,
    text_data,
    source_language,
    target_language,
    max_length
)
```

- Parameters:
  - ```device_id```: To run on GPU, specify the valid cuda device ID. Otherwise, 
  you can provide NULL for this parameter.
  - ```bucketfs_conn```: The BucketFS connection name.
  - ```sub_dir```: The directory where the model is stored in the BucketFS.
  - ```model_name```: The name of the model to use for prediction. You can find the 
  details of the models in [huggingface models page](https://huggingface.co/models).
  - ```text_data```: The text to translate.
  - ```source_language```: The language of the input. Might be required for multilingual models. 
  It does not have any effect for single pair translation models (see [Transformers Translation API](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.TranslationPipeline.__call__)). 
  - ```target_language```:  The language of the desired output. Might be required for multilingual models. 
  It does not have any effect for single pair translation models (see [Transformers Translation API](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.TranslationPipeline.__call__)).
  - ```max_length```: The maximum total length of the translated text. 

The inference results are presented with _TRANSLATION_TEXT_ column, 
combined with the inputs used when calling this UDF. In case of any error during
model loading or prediction, these new columns are set to `null`, and 
column _ERROR_MESSAGE_ is set to the stacktrace of the error. For example:

| BUCKETFS_CONN | SUB_DIR | MODEL_NAME | TEXT_DATA | SOURCE_LANGUAGE | TARGET_LANGUAGE | MAX_LENGTH | TRANSLATION_TEXT | ERROR_MESSAGE |
|---------------|---------|------------|-----------|-----------------|-----------------|------------|------------------|---------------|
| conn_name     | dir/    | model_name | context   | English         | German          | 100        | kontext          | None          |
| ...           | ...     | ...        | ...       | ...             | ...             | ...        | ...              | ...           |


### Zero-Shot Text Classification UDF
This UDF simply provide  the task of predicting a class that wasn't seen by the 
model during training. The UDF takes candidate labels as a comma-separated 
string, and generate probability scores prediction for each label.

```sql
SELECT TE_ZERO_SHOT_TEXT_CLASSIFICATION_UDF(
    device_id,
    bucketfs_conn,
    sub_dir,
    model_name,
    text_data,
    candidate_labels
)
```

- Parameters:
  - ```device_id```: To run on GPU, specify the valid cuda device ID. Otherwise, 
  you can provide NULL for this parameter.
  - ```bucketfs_conn```: The BucketFS connection name.
  - ```sub_dir```: The directory where the model is stored in the BucketFS.
  - ```model_name```: The name of the model to use for prediction. You can find the 
  details of the models in [huggingface models page](https://huggingface.co/models).
  - ```text_data```: The text to be classified.
  - ```candidate labels```: Labels where the given text is classified. Labels 
  should be comma-separated, e.g., `label1,label2,label3`.
  
The inference results are presented with predicted _LABEL_, _SCORE_ and _RANK_ 
columns, combined with the inputs used when calling this UDF. In case of any 
error during model loading or prediction, these new  columns are set to `null`, 
and column _ERROR_MESSAGE_ is set to the stacktrace of the error. For example:

| BUCKETFS_CONN | SUB_DIR | MODEL_NAME | TEXT_DATA | CANDIDATE LABELS | LABEL  | SCORE | RANK | ERROR_MESSAGE |
| ------------- |---------|------------|-----------|------------------|--------|-------|------|---------------|
| conn_name     | dir/    | model_name | text      | label1,label2..  | label1 | 0.75  | 1    | None          |
| conn_name     | dir/    | model_name | text      | label1,label2..  | label2 | 0.70  | 2    | None          |
| ...           | ...     | ...        | ...       | ...              | ...    | ...   | ..   | ...           |  
