# User Guide

## Introduction

The Transformers Extension provides a Python library with UDFs that allow the use of pre-trained NLP models provided by the [Transformers API](https://huggingface.co/docs/transformers/index).

The extension provides two types of UDFs:

* Utility UDFs: UDFs which deal with installation and deletion of pretrained Transformers models in the Exasol BucketFS.
* Prediction UDFs: These are a group of UDFs for each supported task. Each of them uses the downloaded pre-trained model and performs prediction. These are the supported tasks:

   1. AI Custom Classify Extended
   2. AI Entailment Extended
   3. AI Answer Extended
   4. AI Fill Mask Extended
   5. AI Complete Extended
   6. AI Extract Extended
   7. AI Translate Extended
   8. AI Classify Extended

    
This Exasol Extension provides UDFs for interacting with Hugging Face's Transformers API to use pre-trained models on an Exasol cluster.

User Defined Functions (UDFs) are scripts in various programming languages that can be executed in the Exasol database. They can be used by a user for more flexibility in data processing. With the Transformers Extension, we provide multiple UDFs for you to use on your Exasol database. You can find more detailed documentation on UDFs on the [UDF Scripts page](https://docs.exasol.com/db/latest/database_concepts/udf_scripts.htm).

UDFs and the necessary [Script language Container](https://docs.exasol.com/db/latest/database_concepts/udf_scripts/adding_new_packages_script_languages.htm) are stored in Exasol's file system BucketFS, and we also use this to store the Hugging Face models on the Exasol cluster.

More information on the BucketFS can be found at [docs.exasol.com](https://docs.exasol.com/db/latest/database_concepts/bucketfs/bucketfs.htm).

For prerequisites and the setup-guide, please visit the [Getting started](setup.md) file.

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

