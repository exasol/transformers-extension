
# Manage Models in the BucketFS

For managing transformers models in the BucketFS, we provide ways for uploading a model, 
deleting a model, and listing existing models.

### Table of Contents


* [Store Models in BucketFS](#store-models-in-bucketfs)
  * [Model Downloader UDF](#model-downloader-udf)
    * [Name Server](#name-server)
    * [Running the UDF](#running-the-udf)
    * [Selecting the Task Type](#selecting-the-task-type)
  * [Model Uploader Script](#model-uploader-script)
    * [Installation via a Python Function](#installation-via-a-python-function)
* [Delete Models from the BucketFS](#delete-models-from-the-bucketfs)
  * [Delete Model UDF](#delete-model-udf)
  * [Delete model via a Python Function](#delete-model-via-a-python-function)
* [List Models UDF](#list-models-udf)



## Store Models in BucketFS

Before you can use pre-trained models, the models must be stored in the BucketFS.

There are two options to download a Hugging Face transformers model and upload it to the BucketFS of an Exasol database.

|                                                        | [Model Downloader UDF](#model-downloader-udf) | [Model Uploader Script](#model-uploader-script) |
|--------------------------------------------------------|-----------------------------------------------|-------------------------------------------------|
| Convenience                                            | High                                          | Low                                             |
| Exasol database must be enabled to access the internet | Yes                                           | No                                              |
| Execution                                              | UDF inside the database                       | Python script outside the database              |
| Temporary storage                                      | UDF's file system                             | Local file system outside the database          |

The Uploader Script is required in case you do not want to connect your Exasol database directly to the internet. Otherwise, the UDF is more convenient.

Note that the extension currently only supports the `PyTorch` framework. Please make sure that the selected models are in the `Pytorch` model library section.

### Model Downloader UDF

Using the `TE_MODEL_DOWNLOADER_UDF` below, you can download the desired model from the Hugging Face hub and upload it to the BucketFS.

This requires the Exasol database to have internet access, since the UDF will download the model from Hugging Face to the database without saving it somewhere else intermittently.

#### Name Server

If you are using the Exasol DockerDB or an Exasol version 8 setup via [c4](https://docs.exasol.com/db/latest/administration/on-premise/admin_interface/c4.htm), this is not set by default, and you need to specify a name server. For example, setting `nameserver = 8.8.8.8` will use Google's DNS.

You will need to use [ConfD](https://docs.exasol.com/db/latest/confd/confd.htm) to perform this setup. Specifically, you should use the [general_settings](https://docs.exasol.com/db/latest/confd/jobs/general_settings.htm) command.

If you are using the [Integration Test Docker Environment](https://github.com/exasol/integration-test-docker-environment) to control the DockerDB, you can set the name server by passing in: `--nameserver 8.8.8.8`.

#### Running the UDF

Once you have internet access, run the UDF with:

```sql
SELECT TE_MODEL_DOWNLOADER_UDF(
    model_name,
    task_type,
    sub_dir,
    bucketfs_conn,
    token_conn
)
```

[Common Parameters](#common-udf-parameters)
* `model_name`
* `task_type`
* `sub_dir`
* `bucketfs_conn`

Specific parameters
* `token_conn`: The connection name containing the token required for private models. You can use an empty string ('') for public models. For details on how to create a connection object with token information, please check the [Getting Started](#getting-started) section.
* `task_type`: See below.

#### Selecting the Task Type

Some models can be used for multiple types of tasks, but Hugging Face Transformers stores different metadata depending on the task of the model, which affects how the model is loaded later. Setting an incorrect task type, or leaving the task type empty may affect the models performance severely.

Available task types are the same as the names of our available UDFs, namely:
* `filling_mask`
* `question_answering`
* `sequence_classification`
* `text_generation`
* `token_classification`
* `translation`
* `zero_shot_classification`

### Model Uploader Script

You can invoke the Python script below to download the transformer models from the Hugging Face hub to the local filesystem and upload it to the BucketFS.

```shell
python -m exasol_transformers_extension.upload_model <options>
```

For information about the available options common to all Exasol extensions, please refer to the [documentation][pec-user-guide] in the Exasol Python Extension Common package.

In addition, this command provides the following options:

| Option name    | Comment                                                         |
|----------------|-----------------------------------------------------------------|
| `--model-name` | Name of the model, as seen in the Hugging Face hub              |
| `--task-type`  | See the explanations below                                      |
| `--sub-dir`    | Sub-directory in the BucketFS where this model should be stored |
| `--token`      | The [Hugging Face token](#huggingface-token), if required       |

`--task-type` specifies the type of task for which you plan to use the model, see [Selecting the Task Type](#selecting-the-task-type).

#### Installation via a Python Function

Alternatively, you can install a Hugging Face model using a Python function instead of a shell command.

Function
`exasol_transformers_extension.utils.model_utils.install_huggingface_model()` expects the following arguments
* A BucketFS location
* Argument `model_spec` of type `BucketFSModelSpecification` containing
  * `model_name`
  * `task_type`
  * `sub_dir`
* An optional `model_factory`
* An optional `tokenizer_factory`
* An optional `huggingface_token`

**Please note**:
* The former function `exasol_transformers_extension.upload_model.upload_model_to_bfs_location()` is now deprecated and internally now also uses the function `install_huggingface_model()` described above.
* The former function returned type `Path`, while the new implementation returns type `bfs.path.PathLike`.

## Delete Models from the BucketFS

Similar to [Store Models in BucketFS](#store-models-in-bucketfs), you have two options to delete an uploaded model from BucketFS:
- via a UDF call. This is more convenient as you can call it via SQL.
- via a Python API call. This is useful when you have an automated toolchain, e.g. a CI build.
In order to do this, you might need to find out which models are safed in the Exasol BucketFS. To do this, 
we provide the `TE_LIST_MODELS_UDF`. See details at the end of this section.


### Delete Model UDF

Using the `TE_DELETE_MODEL_UDF` below, you can delete a model from BucketFS. The parameter values are similar to that one used in [Store Models in BucketFS](#store-models-in-bucketfs).

Run the UDF with:

```sql
SELECT TE_DELETE_MODEL_UDF(
    bfs_conn,
    sub_dir,
    model_name,
    task_type
)
```

See [Common Parameters](#common-udf-parameters) for information about `bfs_conn`, `sub_dir` and `model_name`. All values, including the [Task Type](#selecting-the-task-type), should have the same value as used during the model installation.

Additional output columns
* success: True if deletion was successful, False otherwise
* error_message: None if deletion was successful, a string containing detailed error information otherwise

Example output:

| BUCKETFS_CONN | SUB_DIR | MODEL_NAME   | TASK_TYPE   | SUCCESS | ERROR_MESSAGE  |
| ------------- |---------|--------------|-------------|---------|----------------|
| conn_name     | dir/    | model_name_a | task_type_a | False   | file not found |
| conn_name     | dir/    | model_name_a | task_type_b | True    |                |
| ...           | ...     | ...          | ...         | ...     | ...            |


### Delete model via a Python Function

Alternatively, you can delete a Hugging Face model using a Python function instead of the UDF.

Function
`exasol_transformers_extension.utils.model_utils.delete_model()` expects the following arguments
* A BucketFS location
* Argument `model_spec` of type `BucketFSModelSpecification` containing
  * `model_name`
  * `task_type`
  * `sub_dir`

The function will raise an exception if the parameters refer to a none-existing file in BucketFS.

## List Models UDF

The `TE_LIST_MODELS_UDF` lists all the models installed with the Transformers Extension in a given directory in the BucketFS.
It takes a BucketFS connection and a director as input, and will return a list of models found in thet directory.
The output will contain the `model_name`, `task_type` and path of the model in the BucketFS, as well as a column 
for potential error messages, in addition to the input.

[Common Parameters](#common-udf-parameters)
* `bucketfs_conn`
* `sub_dir`

This UDF will fail to return a model if it was saved with the sub_dir parameter empty, 
or if no config.json file can be found in the model files.

Call the UDF like this:

```sql
SELECT TE_LIST_MODELS_UDF(
    bucketfs_conn,
    sub_dir,
)
```
Example Output:

| BUCKETFS_CONN | SUB_DIR | MODEL_NAME | TASK_NAME | MODEL_PATH               | ERROR_MESSAGE |
|---------------|---------|------------|-----------|--------------------------|---------------|
| conn_name     | dir/    | model_name | task_name | dir/model_name_task_name |  None         |
| ...           | ...     | ...        | ...       | ...                      |  ...          |

