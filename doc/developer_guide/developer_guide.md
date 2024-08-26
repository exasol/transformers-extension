# Developer Guide


In this developer guide we explain how to build this project and how you can add 
new transformer tasks and tests.


## Installation
There are two ways to install the Transformers Extension Package:

### 1. Build and install the Python Package
This project needs Python 3.10 or above installed on the development machine. 
In addition, in order to build Python packages you need to have the [Poetry](https://python-poetry.org/)
(>= 1.1.11) package manager. Then you can install and build the `transformers-extension` as follows:
```bash
poetry install
poetry build
```

### 2. Download and install the pre-build wheel
Instead of building yourself, the latest version of the Python package of this extension can be downloaded 
from the Releases in the GitHub Repository (see [the latest release](https://github.com/exasol/transformers-extension/releases/latest)).
Please download the built archive 
`exasol_transformers_extension-<version-number>-py3-none-any.whl`(`transformers_extension.whl` in older versions) 
and install it as follows:
```bash
pip install <path/wheel-filename.whl> --extra-index-url https://download.pytorch.org/whl/cpu
```

### Check wheel installation

The wheel should be installed in `transformers-extension/dist`. After updating and building a new release 
there may be multiple wheels installed here. This leads to problems, so check and delete the old wheels if necessary.
You may also need to check 
`transformers-extension/language_container/exasol_transformers_extension_container/flavor_base/release/dist` for the same reason.

### Run Tests
All unit and integration tests can be run within the Poetry environment created 
for the project using nox. See [the nox file](../../noxfile.py) for all tasks run by nox. There are three tasks for tests.

Run unit tests:
```bash
      poetry run nox -s unit_tests
```
Start a test database and run integration all tests:
```bash
      poetry run nox -s start_database
      poetry run nox -s integration_tests
```
run parts of the integration tests:
```bash
      poetry run nox -s onprem_integration_tests
      poetry run nox -s saas_integration_tests
      poetry run nox -s without_db_integration_tests
```
You can find more information regarding the tests in the [Tests](#tests) section below

## Add Transformer Tasks
In the transformers-extension library, the 8 most popular NLP tasks provided by 
[Transformers API](https://huggingface.co/docs/transformers/index) have already 
been defined. We created separate UDF scripts for each NLP task. You can find 
these tasks and UDF script usage details in the [User Guide](../user_guide/user_guide.md#prediction-udfs).  
This section shows you step by step how to add a new NLP task to this library.

### 1. Add a UDF Template
The new task's UDF template should be added to the `exasol_transformers_extension/resources/templates/` 
directory. Please pay attention that the UDF script is uses _"SET UDF"_  and the inputs 
are received ordered by pre-determined columns. In addition, the first 4 input 
arguments of the UDF script should be:

  - ```device_id```: To run on GPU, specify the valid cuda device ID. Otherwise, 
  you can provide NULL for this parameter.
  - ```bucketfs_conn```: The BucketFS connection name 
  - ```sub_dir```: The directory where the model is stored in the BucketFS.
  - ```model_name```: The name of the model to use for prediction. You can find the 
  details of the models in [huggingface models page](https://huggingface.co/models).

Please note that the output emitted by the UDF is created by adding the model 
inference output to the inputs.

### 2. Define UDF Caller
Before implementing the UDF logic (examined in item 4 in this section), the 
`run` function responsible for calling the newly created UDF script should be 
defined in `exasol_transformers_extension/udfs/callers/`.

### 3. UDF Template-Caller Matching 
The added UDF template and defined UDF caller should be added to the dictionary
in the `exasol_transformers_extension/deployment/constants.py` script. Thus, 
we know which template belongs to which script during deployment.

### 4. Implement Task Logic in UDF Script
The UDF class, in which we implement the logic of the desired task, must be 
defined under the `exasol_transformers_extension/udfs/models/` directory. This 
class should extend the _BaseModelUDF_ class. Moreover, new output columns 
expected from this task should be specified in the `new_columns` list.

`BaseModelUDF` contains common operations for all task UDFs. For example:
- accesses data in batches with predefined batch size
- manages the script cache
- reads the corresponding model from BucketFS into cache
- creates model pipeline through transformer api
- manages the creation of model predictions and the preparation of results.


Users should implement the following methods in the UDF class 
that extends the `BaseModel UDF`:
 - `extract_unique_param_based_dataframes` : Even if the data in a given 
dataframe all have the same model, there might be differences within the given 
dataframe with different model parameters (e.g. _top_k_ parameter in [FillingMaskUDF](../../exasol_transformers_extension/udfs/models/filling_mask_udf.py)). 
This method is responsible for extracting unique dataframes which share both the
same model and model parameters.
 - `execute_prediction` : Performs prediction on a given text list using 
recently loaded models.
- `create_dataframes_from_predictions` : Converts list of predictions to 
pandas dataframe.
- `append_predictions_to_input_dataframe`: Reformats the dataframe used in 
prediction, such that each input row has a row for each prediction result.
 


## Tests

#### 1. Unit Tests
- Unit tests use the [udf-mock-python](https://github.com/exasol/udf-mock-python) 
library that tests UDFs locally without a database. 
- Different scenarios with  different UDF inputs and different model parameters 
are defined under the `tests/unit_tests/udf_wrapper_params/` directory. 
- These different scenarios are parameterized in the UDF [tests](../../tests/unit_tests/udfs).

#### 2. Integration Tests
These tests are grouped into two groups and there are separate tests for each 
UDF script in each group:
- `without db` tests the UDF class and functionality that includes the UDF logic.
- `with_db` performs end-to-end test by running the UDF query statements in the database. 

The automatic run of the Integration tests on GitHub push are moved into AWS for this repository. They are 
only run if you add `[CodeBuild]` to the commit message.
Currently, the CodeBuild project is managed manually and is triggered with a webhook on branch push.
For this our aws-ci user is added to this Repository. The webhook can be configured in the AWS CodeBuild 
project directly.
The CodeBuild project also uses our DockerHub user for the build. For this it has access to the AWS SecretsManager.


## Good to know

* Hugging Face models consist of 2 parts, the model and the Tokenizer. 
Most of our functions deal with both parts