
# User Guide

The Transformers Extension provides a Python library with UDFs that allow the 
use of pre-trained NLP models provided by the [Transformers API](https://huggingface.co/docs/transformers/index).

The extension provides two types of UDFs:
 - DownloaderUDF :  It is responsible to download the specified pre-defined model into the Exasol BucketFS.
 - Prediction UDFs: These are a group of UDFs for each supported task. Each of them uses the downloaded the pre-trained model and perform the prediction. The supported tasks:
   1. Sequence Classification for Single Text 
   2. Sequence Classification for Text Pair
   3. Question Answering
   4. Masked Language Modelling
   5. Text Generation
   6. Token Classification
   7. Translation


## Table of Contents

- [Getting Started](#getting-started)
- [Setup](#setup)
- [Model Downloader UDF](#model-downloader-udf)
- [Prediction UDFs](#prediction_udfs)
  1. [Sequence Classification for Single Text UDF](#sequence_classification_for_single_text_udf)
  2. [Sequence Classification for Text Pair UDF](#sequence_classification_for_text_pair_udf)
  3. [Question Answering](#question_answering)



## Getting Started
- Exasol DB
  - The Exasol cluster must already be running with version 7.1 or later.
  - DB connection information and credentials are needed.
- BucketFS Connection 
  - An Exasol connection object must be created with Exasol BucketFS connection 
  information and credentials. 
  - An example connection object is created 
  as follows. For more information please check the [Create Connection in Exasol](https://docs.exasol.com/sql/create_connection.htm?Highlight=connection) document:  
  ```buildoutcfg
  CREATE OR REPLACE CONNECTION <BUCKETFS_CONNECTION_NAME>
      TO '<BUCKETFS_ADDRESS>'
      USER '<BUCKETFS_USER>'
      IDENTIFIED BY '<BUCKETFS_PASS>'
  
## Setup
### The Built Archive
#### Download The Built Archive
- The latest version of the packaged built archive of this extension can be 
downloaded from the Releases in Github Repository 
(see [the latest release](https://github.com/exasol/transformers-extension/releases/latest)).
Please download the following built archive:
```buildoutcfg 
transformers_extension.whl
```

#### Install The Built Archive
- Install the packaged transformers-extension project as follows:
```buildoutcfg
pip install transformers_extension.whl
```

### The Pre-built Language Container
#### Download Language Container
- In order to get this extension run,  the language container of this extension is required.
Please download the language container from the Releases in Github Repository. 
(see [the latest release](https://github.com/exasol/transformers-extension/releases/latest)).
 
#### Install Language Container
- To install the language container, it is necessary to load the container in BucketFS 
and register it to the database. The following command provides this setup:
```buildoutcfg
python -m transformers_extension.main language-container
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
    --container-file <path/to/container-file.tar.gz>       
```

### Deployment
- Deploy all necessary scripts installed in the previous step to the specified 
`SCHEMA` in Exasol DB with the same `LANGUAGE_ALIAS`  using the following python cli command:
```buildoutcfg
python -m transformers_extension.main scripts
    --dsn <DB_HOST:DB_PORT> \
    --db-user <DB_USER> \
    --db-pass <DB_PASSWORD> \
    --schema <SCHEMA> \
    --language-alias <LANGUAGE_ALIAS>
```

## Model Downloader UDF
Before using pre-trained models, the models must be downloaded and cached into 
BucketFS. For this, you can use the `TE_MODEL_DOWNLOADER_UDF` script as follows
```buildoutcfg
SELECT TE_MODEL_DOWNLOADER_UDF(
    model_name,
    sub_dir,
    bucketfs_conn
)
```
- Parameters:
  - ```model_name```: The name of the model to be downloaded. You can find the 
  details and names of the models [here](https://huggingface.co/models).
  - ```sub_dir```: The directory where the model is downloaded in the cache.
  - ```bucketfs_conn```: The BucketFS connection name 

Note that the extension currently only supports the `PyTorch` framework. 
Please make sure that the selected models are in the `Pytorch` library.

## Prediction UDFs
We provided 7 prediction UDFs, each performing an NLP task through the [transformers API](https://huggingface.co/docs/transformers/task_summary). 
These tasks use models cached in BucketFS to make an inference on user-provided inputs.

### Sequence Classification for Single Text UDF
This UDF classifies the given single text  according to a given number of 
classes of the specified  model. Example usage is given below:
```buildoutcfg
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
  you can leave this field blank.
  - ```bucketfs_conn```: The BucketFS connection name 
  - ```sub_dir```: The directory where the model is downloaded in the cache.
  - ```model_name```: The name of the model to be downloaded. You can find the 
  details and names of the models [here](https://huggingface.co/models).
  - ```text_data```: The input text to be classified

The inference results are presented as predicted _LABEL_ and confidence 
 _SCORE_ columns, by combining with the inputs used when calling 
this UDF. For example:

    | BUCKETFS_CONN | SUB_DIR | MODEL_NAME | TEXT_DATA | LABEL | SCORE |
    | ------------- | ------- | ---------- | --------- | ----- | ----- |
    | conn_name     | dir/    | model_name | text      | L1    | 0.75  |
    | ...           | ...     | ...        | ...       | ...   | ...   |

### Sequence Classification for Text Pair UDF
This UDF takes two input sequences and compares them, e.g., it is used to 
determine if two sequences are paraphrases of each other. Example usage is given below:

```buildoutcfg
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
  you can leave this field blank.
  - ```bucketfs_conn```: The BucketFS connection name 
  - ```sub_dir```: The directory where the model is downloaded in the cache.
  - ```model_name```: The name of the model to be downloaded. You can find the 
  details and names of the models [here](https://huggingface.co/models).
  - ```first_text```: The first input text
  - ```second_text```: The second input text

The inference results are presented as predicted _LABEL_ and confidence 
 _SCORE_ columns, by combining with the inputs used when calling this UDF.

### Question Answering UDF
This UDF extracts answer(s) from a given question text. With the `top_k` 
input parameter, up to `k` answers with the best inference scores can be achieved. 
Example usage is given below:
```buildoutcfg
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
  you can leave this field blank.
  - ```bucketfs_conn```: The BucketFS connection name 
  - ```sub_dir```: The directory where the model is downloaded in the cache.
  - ```model_name```: The name of the model to be downloaded. You can find the 
  details and names of the models [here](https://huggingface.co/models).
  - ```question```: The question text
  - ```context_text```: The context text, associated with question
  - ```top_k```: The number of answers to return. Note that, `k` number of answers are not guaranteed. If there are not enough options 
in the context, it might return less than `top_k` answers (see the [explanation](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.QuestionAnsweringPipeline.__call__.topk)).

The inference results are presented as predicted _ANSWER_ and confidence 
 _SCORE_ columns, by combining with the inputs used when calling this UDF.
If `top_k` > 1, each input row is repeated for each answer. For example:

    | BUCKETFS_CONN | SUB_DIR | MODEL_NAME | QUESTION | CONTEXT | TOP_K | ANSWER | SCORE |
    | ------------- | ------- | ---------- | -------- | ------- | ----- | ------ | ----- |
    | conn_name     | dir/    | model_name | q1       | c1      | 2     | a1     | 0.75  |
    | conn_name     | dir/    | model_name | q1       | c1      | 2     | a2     | 0.70  |
    | ...           | ...     | ...        | ...      | ...     | ...   | ...    | ...   |
   


