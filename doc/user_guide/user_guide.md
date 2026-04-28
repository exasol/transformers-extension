# User Guide

## Introduction

The Transformers Extension provides a Python library with UDFs that allow the use of pre-trained NLP models provided by the [Transformers API](https://huggingface.co/docs/transformers/index).

The extension provides two types of UDFs:

* Utility UDFs: Managing of Transformers models in the Exasol BucketFS.
* Prediction UDFs: Use a downloaded pre-trained model to perform prediction.

### A Note on UDFs :

User Defined Functions (UDFs) are scripts in various programming languages that can be executed in the Exasol database. They can be used by a user for more flexibility in data processing. With the Transformers Extension, we provide multiple UDFs for you to use on your Exasol database. You can find more detailed documentation on UDFs on the [UDF Scripts page](https://docs.exasol.com/db/latest/database_concepts/udf_scripts.htm).

UDFs and the necessary [Script language Container](https://docs.exasol.com/db/latest/database_concepts/udf_scripts/adding_new_packages_script_languages.htm) are stored in Exasol's file system BucketFS, and we also use this to store the Hugging Face models on the Exasol cluster.

More information on the BucketFS can be found at [docs.exasol.com](https://docs.exasol.com/db/latest/database_concepts/bucketfs/bucketfs.htm).

For prerequisites and the setup-guide, please visit the [Getting started](setup.md) file.

### Utility UDFs

This Exasol Extension provides UDFs for interacting with Hugging Face's Transformers API to use pre-trained models on an Exasol cluster.

These UDFs deal with the installation and deletion of pretrained Transformers models in the Exasol BucketFS.

We provide the following UDFs:

| UDF Name                     | Use                                                            |
|------------------------------|----------------------------------------------------------------|
| TE_MODEL_DOWNLOADER_UDF      | Download a specific model from Hugging Face.                   |
| TE_LIST_MODELS_UDF           | List all models available in the BucketFS.                     |
| INSTALL_AI_DEFAULT_MODEL_UDF | Install all default models used in the Transformers Extension. |
| TE_DELETE_MODEL_UDF          | Delete a specific model from the BucketFS.                     |

You can find further information on these UDFs [here](manage_models.md).


### Prediction UDFs

These UDFs call a model stored in the BucketFS, and use it to make predictions on the given input data.

We have selected a curated list of models, which are used in our UDFs. 
These UDFs require only minimal configuation to use:


| UDF Name      | task_type            | Use                                                                  |
|---------------|----------------------|----------------------------------------------------------------------|
| AI Sentiment  | text-classification  | Classifies the given text according the sentiment found in the text. |

However, if you want to configure a task to your specific needs, 
UDFs with the suffix "Extended" in the name allow you to specify all available 
parameters for each input row. 

For example, you may want to select a specific model to be used. 

These are the available UDFs:

| UDF Name                    | task_type                | Use                                                                  |
|-----------------------------|--------------------------|----------------------------------------------------------------------|
| AI Custom Classify Extended | text-classification      | Classifies the given text into classes known to the model.           |
| AI Entailment Extended      | text-classification      | Takes two input texts and compares them.                             |
| AI Answer Extended          | question-answering       | Extracts answer(s) from a given question text.                       |
| AI Fill Mask Extended       | fill-mask                | Replace ```<mask>``` tokens in the input with predicted text.        |
| AI Complete Extended        | text-generation          | Predict the continuation of the given text.                          |
| AI Extract Extended         | token-classification     | Find and lable tokens in a given text.                               |
| AI Translate Extended       | translation              | This UDF translates a given text from one language to another.       |
| AI Classify Extended        | zero-shot-classification | This UDF classifies the input text into classes defined by the user. |

Each UDF uses models for a predefined Transformers task. 
Models which do not support this task will perform poorly. So take care to match your 
selected model to the UDF for your desired task.

More details on the available parameters can be found in [detailed documentation](invoke_models.md).


## Common UDF Parameters

Many UDFs use a set of common parameters:

* `device_id`: To run on a GPU, specify the valid cuda device ID. Value `NULL` means to use the CPU instead.
* `bucketfs_conn`: The BucketFS connection name.
* `sub_dir`: The directory where the model is stored in the BucketFS.
* `model_name`: The name of the model to use for prediction. You can find the details of the models on the [Hugging Face models page](https://huggingface.co/models).

## Common Output Behavior

Each of the UDFs generates an output containing the original input columns passed to the UDF plus additional columns containing the inference results.

In case of any error during model loading or prediction, the additional output columns are set to `NULL` and column _ERROR_MESSAGE_ is set to the stacktrace of the error.

## What's Next:

For information on managing the models, please visit the [Manage Models in the BucketFS](manage_models.md) guide.

For information on invoking the models, please visit the guide on [Using Prediction UDFs](invoke_models.md).

