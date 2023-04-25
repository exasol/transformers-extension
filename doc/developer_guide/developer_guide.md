# Developer Guide


In this developer guide we explain how you can build this project, how to add 
new transformer tasks and tests.


## Building the Project

### 1. Build the Python Package
This project needs python interpreter Python 3.8 or above installed on the 
development machine. In addition, in order to build python packages you need to 
have >= [poetry](https://python-poetry.org/) 1.1.11 package manager. Then you can 
install and build as follows:
```bash
poetry install
poetry build
```

### 2. Install the Project
The latest version of the python package of this extension can be downloaded 
from the Releases in GitHub Repository (see [the latest release](https://github.com/exasol/transformers-extension/releases/latest)).
Please download the built archive `transformers_extension.whl` and install as follows:
```bash
pip install dist/transformers_extension.whl
```

### 3. Run All Tests
All unit and integraiton tests can be run within the poetry environment created 
for the project as follows:
```bash
poetry run pytest tests
```


## Add Transformer Tasks
In the transformers-extension library, the 8 most popular NLP tasks provided by 
[Transformers API](https://huggingface.co/docs/transformers/index) have already 
been defined. We created separate UDF scripts for each NLP task. You can find 
these tasks and UDF script usage details in the [User Guide](../user_guide/user_guide.md#prediction-udfs).  
This section shows you step by step how to add a new NLP task to this library.

### 1. Add UDF Template
The new task's UDF template should be added to the `exasol_transformers_extension/resources/templates/` 
directory. Please pay attention that the UDF script is SET UDF and the inputs 
are received ordered by pre-determined columns. In addition, the first 4 input 
arguments of the UDF script should be:

  - ```device_id```: To run on GPU, specify the valid cuda device ID. Otherwise, 
  you can provide NULL for this parameter.
  - ```bucketfs_conn```: The BucketFS connection name 
  - ```sub_dir```: The directory where the model is stored in the BucketFS.
  - ```model_name```: The name of the model to use for prediction. You can find the 
  details of the models in [huggingface models page](https://huggingface.co/models).
  - ```text_data```: The input text to be classified

Please note that, the output emitted by the UDF is created by adding the model 
inference output to the inputs.

### 2. Define UDF Caller
Before implementing the UDF logic (examined in item 4 in this section), the 
`run` function responsible for calling the newly created UDF script should be 
defined in `exasol_transformers_extension/udfs/callers/`

### 3. UDF Template-Caller matching 
The added UDF template and defined UDF caller should be added to the dictionary
in the `exasol_transformers_extension/deployment/constants.py` script. Thus, 
we are able to be aware of which template belongs to which script during deployment.

### 4. Implement Task Logic in UDF script
The UDF class, in which we implement the logic of the desired task, must be 
defined under the `exasol_transformers_extension/udfs/models/` directory. This 
class should extend the _BaseModelUDF_ class. Moreover, new output columns 
expected from this task should be specified in the `new_columns` list.

`BaseModelUDF` contains common operations for all task UDFs. For example:
- accesses data part-by-part based on predefined batch size
- manages the script cache
- reads the corresponding model from BucketFS into cache
- creates model pipeline through transformer api
- manages the creation of model predictions and the preparation of results.


Users should implement the following methods in the UDF class 
that extends the `BaseModel UDF`:
 - `extract_unique_param_based_dataframes` : Even if the data in a given 
dataframe all have the same model, there might be differences within the given 
dataframe with different model parameters (e.g. _top_k_ parameter in [FillingMaskUDF](../../exasol_transformers_extension/udfs/models/filling_mask_udf.py)). 
This method is responsible for extracting unique dataframes that shares both 
same model and model parameters.
 - `execute_prediction` : Performs prediction on a given text list using 
recently loaded models.
- `create_dataframes_from_predictions` : Converts list of predictions to 
pandas dataframe.
- `append_predictions_to_input_dataframe`: Reformats the dataframe used in 
prediction, such that each input rows has a row for each prediction results.
 


## Tests

#### 1. Unit Tests
UDF Mock, for different parameters different scenarios are defined as udf_wrappers and tests are paramaterized with these scenarios

#### 2. Integration Tests
without db, functionality , with db end-to end test.


