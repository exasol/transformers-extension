
# User Guide

The Transformers Extension provides a Python library with UDFs that allow the 
use of pre-trained NLP models provided by the [Transformers API](https://huggingface.co/docs/transformers/index).

The extension provides two types of UDFs:
 - DownloaderUDF :  It is responsible to download the specified pre-defined model into the Exasol BucketFS.
 - Prediction UDFs: These are a group of UDFs for each supported task. Each of them uses the downloaded pre-trained model and perform prediction. These supported tasks:
   1. Sequence Classification for Single Text 
   2. Sequence Classification for Text Pair
   3. Question Answering
   4. Masked Language Modelling
   5. Text Generation
   6. Token Classification
   7. Text Translation


## Table of Contents

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



## Getting Started
- Exasol DB
  - The Exasol cluster must already be running with version 7.1 or later.
  - DB connection information and credentials are needed.
- BucketFS Connection 
  - An Exasol connection object must be created with Exasol BucketFS connection 
  information and credentials. 
  - An example connection object is created as follows: 
  ```sql
  CREATE OR REPLACE CONNECTION <BUCKETFS_CONNECTION_NAME>
      TO '<BUCKETFS_ADDRESS>'
      USER '<BUCKETFS_USER>'
      IDENTIFIED BY '<BUCKETFS_PASS>'
  ```
  - The `BUCKETFS_ADDRESS` looks like the following:
  ```buildoutcfg
    http[s]://<BUCKETFS_HOST>:<BUCKETFS_PORT>/<BUCKET_NAME>/<PATH_IN_BUCKET>;<BUCKETFS_NAME>
  ```
  - For more information please check the [Create Connection in Exasol](https://docs.exasol.com/sql/create_connection.htm?Highlight=connection) document.
  
## Setup
### The Python Package
#### Download The Python Wheel Package
- The latest version of the python package of this extension can be 
downloaded from the Releases in GitHub Repository 
(see [the latest release](https://github.com/exasol/transformers-extension/releases/latest)).
Please download the following built archive:
```buildoutcfg 
transformers_extension.whl
```

#### Install The Python Wheel Package
- Install the packaged transformers-extension project as follows:
```shell
pip install transformers_extension.whl
```

### The Pre-built Language Container

This extension requires the installation of the language container for this 
extension to run. It can be installed in two ways:Quick and Customized 
installations

#### Quick Installation
The desired language container is downloaded and installed by executing the 
deployment script below with the desired version. (see GitHub Releases 
[the latest release](https://github.com/exasol/transformers-extension/releases).

  ```buildoutcfg
  python -m exasol_transformers_extension.deploy language-container
      --dsn <DB_HOST:DB_PORT> \
      --db-user <DB_USER> \
      --db-pass <DB_PASSWORD> \
      --bucketfs-name <BUCKETFS_NAME> \
      --bucketfs-host <BUCKETFS_HOST> \
      --bucketfs-port <BUCKETFS_PORT> \
      --bucketfs-user <BUCKETFS_USER> \
      --bucketfs-password <BUCKETFS_PASSWORD> \
      --bucket <BUCKETFS_NAME> \
      --path-in-bucket <PATH_IN_BUCKET> \
      --language-alias <LANGUAGE_ALIAS> \ 
      --version <RELEASE_VERSION>       
  ```

#### Customized Installation
In this installation, you can install the desired or customized language 
container. In the following steps,  it is explained how to install the 
language container file released in GitHub Releases section.


##### Download Language Container
   - The language container is split into parts and then uploaded to GitHub Release section.
   - These parts are named with the `language_container_part_` prefix. 
   - Please download all parts of the language container from the Releases section. 
(see [the latest release](https://github.com/exasol/transformers-extension/releases/latest)).
- Before installing the language container, these parts must be combined using the following command::
```shell
cat language_container_part_* > language_container.tar.gz
```

##### Install Language Container
There are two ways to install the language container: (1) using a python script and (2) manual installation. See the next paragraphs for details.

  1. *Installation with Python Script*

     To install the language container, it is necessary to load the container 
     into the BucketFS and register it to the database. The following command 
     provides this setup using the python script provided with this library:

      ```buildoutcfg
      python -m exasol_transformers_extension.deploy language-container
          --dsn <DB_HOST:DB_PORT> \
          --db-user <DB_USER> \
          --db-pass <DB_PASSWORD> \
          --bucketfs-name <BUCKETFS_NAME> \
          --bucketfs-host <BUCKETFS_HOST> \
          --bucketfs-port <BUCKETFS_PORT> \
          --bucketfs-user <BUCKETFS_USER> \
          --bucketfs-password <BUCKETFS_PASSWORD> \
          --bucket <BUCKETFS_NAME> \
          --path-in-bucket <PATH_IN_BUCKET> \
          --language-alias <LANGUAGE_ALIAS> \ 
          --container-file <path/to/language_container.tar.gz>       
      ```

  2. *Manual Installation*

     In the manual installation, the pre-built container should be firstly 
     uploaded into BucketFS. In order to do that, you can use 
     either a [http(s) client](https://docs.exasol.com/database_concepts/bucketfs/file_access.htm) 
     or the [bucketfs-client](https://github.com/exasol/bucketfs-client). 
     The following command uploads a given container into BucketFS through curl 
     command, an http(s) client: 
      ```shell
      curl -vX PUT -T \ 
          "<CONTAINER_FILE>" 
          "http://w:<BUCKETFS_WRITE_PASSWORD>@<BUCKETFS_HOST>:<BUCKETFS_PORT>/<BUCKETFS_NAME>/<PATH_IN_BUCKET><CONTAINER_FILE>"
      ```

      Please note that specifying the password on command line will make your shell record the password in the history. To avoid leaking your password please consider to set an environment variable. The following examples sets environment variable `BUCKETFS_WRITE_PASSWORD`:
      ```shell 
        read -sp "password: " BUCKETFS_WRITE_PASSWORD
      ```

      The uploaded container should be secondly activated through adjusting 
      the session parameter `SCRIPT_LANGUAGES`. The activation can be scoped
      either session-wide (`ALTER SESSION`) or system-wide (`ALTER SYSTEM`). 
      The following example query activates the container session-wide:

      ```sql
      ALTER SESSION SET SCRIPT_LANGUAGES=\
      <ALIAS>=localzmq+protobuf:///<BUCKETFS_NAME>/<BUCKET_NAME>/<PATH_IN_BUCKET><CONTAINER_NAME>/?\
              lang=<LANGUAGE>#buckets/<BUCKETFS_NAME>/<BUCKET_NAME>/<PATH_IN_BUCKET><CONTAINER_NAME>/\
              exaudf/exaudfclient_py3
      ```
     
      In project transformer-extensions replace `<ALIAS>` by `_PYTHON3_TE_` and `<LANGUAGE>` by `_python_`.
      For more details please check [Adding New Packages to Existing Script Languages](https://docs.exasol.com/database_concepts/udf_scripts/adding_new_packages_script_languages.htm).


### Deployment
- Deploy all necessary scripts installed in the previous step to the specified 
`SCHEMA` in Exasol DB with the same `LANGUAGE_ALIAS`  using the following python cli command:
```buildoutcfg
python -m exasol_transformers_extension.deploy scripts
    --dsn <DB_HOST:DB_PORT> \
    --db-user <DB_USER> \
    --db-pass <DB_PASSWORD> \
    --schema <SCHEMA> \
    --language-alias <LANGUAGE_ALIAS>
```

## Model Downloader UDF
Before you can use pre-trained models, the models must be stored in the 
BucketFS. We provide two different ways to load transformers models 
into BucketFS:

### 1. Model Downloader UDF
Using the `TE_MODEL_DOWNLOADER_UDF` below, you can download the desired model 
from the huggingface hub and upload it to bucketfs.

```sql
SELECT TE_MODEL_DOWNLOADER_UDF(
    model_name,
    sub_dir,
    bucketfs_conn
)
```
- Parameters:
  - ```model_name```: The name of the model to use for prediction. You can find the 
  details of the models in [huggingface models page](https://huggingface.co/models).
  - ```sub_dir```: The directory where the model is stored in the BucketFS.
  - ```bucketfs_conn```: The BucketFS connection name 

Note that the extension currently only supports the `PyTorch` framework. 
Please make sure that the selected models are in the `Pytorch` model library section.


### 2. Upload Model Script
You can invoke the python script as below which allows to load the transformer 
models from the local filesystem into BucketFS:

  ```buildoutcfg
  python -m exasol_transformers_extension.upload_model
      --bucketfs-name <BUCKETFS_NAME> \
      --bucketfs-host <BUCKETFS_HOST> \
      --bucketfs-port <BUCKETFS_PORT> \
      --bucketfs-user <BUCKETFS_USER> \
      --bucketfs-password <BUCKETFS_PASSWORD> \
      --bucket <BUCKETFS_NAME> \
      --path-in-bucket <PATH_IN_BUCKET> \
      --model-name <MODEL_NAME> \
      --subd-dir <SUB_DIRECTORY> \
      --model-path <MODEL_PATH> \
      --tokenizer-path <TOKENIZER_PATH>     
  ```


## Prediction UDFs
We provided 7 prediction UDFs, each performing an NLP task through the [transformers API](https://huggingface.co/docs/transformers/task_summary). 
These tasks cache the model downloaded to BucketFS and make an inference using the cached models with user-supplied inputs.

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
this UDF. For example:

| BUCKETFS_CONN | SUB_DIR | MODEL_NAME | TEXT_DATA | LABEL   | SCORE |
| ------------- | ------- | ---------- | --------- |---------| ----- |
| conn_name     | dir/    | model_name | text      | label_1 | 0.75  |
| ...           | ...     | ...        | ...       | ...     | ...   |


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
in the context, it might return less than `top_k` answers (see the [top_k parameter of QuestoinAnswering](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.QuestionAnsweringPipeline.__call__.topk)).

The inference results are presented with predicted _ANSWER_ and confidence 
 _SCORE_ columns, combined with the inputs used when calling this UDF.
If `top_k` > 1, each input row is repeated for each answer. For example:

| BUCKETFS_CONN | SUB_DIR | MODEL_NAME | QUESTION   | CONTEXT   | TOP_K | ANSWER   | SCORE |
| ------------- | ------- | ---------- |------------|-----------| ----- |----------| ----- |
| conn_name     | dir/    | model_name | question_1 | context_1 | 2     | answer_1 | 0.75  |
| conn_name     | dir/    | model_name | question_2 | context_1 | 2     | answer_2 | 0.70  |
| ...           | ...     | ...        | ...        | ...       | ...   | ...      | ...   |


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


The inference results are presented with _FILLED_TEXT_ and confidence 
 _SCORE_ columns, combined with the inputs used when calling this UDF.
If `top_k` > 1, each input row is repeated for each prediction. For example:

| BUCKETFS_CONN | SUB_DIR | MODEL_NAME | TEXT_DATA     | TOP_K | FILLED_TEXT   | SCORE |
| ------------- | ------- | ---------- |---------------| ----- |---------------| ----- |
| conn_name     | dir/    | model_name | text `<mask>` | 2     | text filled_1 | 0.75  |
| conn_name     | dir/    | model_name | text `<mask>` | 2     | text filled_2 | 0.70  |
| ...           | ...     | ...        | ...           | ...   | ...           | ...   |


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
combined with the inputs used when calling this UDF.


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
For example:

| BUCKETFS_CONN | SUB_DIR | MODEL_NAME | TEXT_DATA | AGGREGATION_STRATEGY | START_POS | END_POS | WORD | ENTITY | SCORE |
| ------------- | ------- | ---------- |-----------|----------------------|-----------|---------|------|--------|-------|
| conn_name     | dir/    | model_name | text      | simple               | 0         | 4       | text | noun   | 0.75  |
| ...           | ...     | ...        | ...       | ...                  | ...       | ...     | ...  | ..     | ...   |



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
combined with the inputs used when calling this UDF. For example:

| BUCKETFS_CONN | SUB_DIR | MODEL_NAME | TEXT_DATA | SOURCE_LANGUAGE | TARGET_LANGUAGE | MAX_LENGTH | TRANSLATION_TEXT |
| ------------- | ------- | ---------- |-----------|-----------------|-----------------|------------| ---------------- |
| conn_name     | dir/    | model_name | context   | English         | German          | 100        | kontext          |
| ...           | ...     | ...        | ...       | ...             | ...             | ...        | ...              |
