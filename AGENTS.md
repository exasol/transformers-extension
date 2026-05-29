# Local Development Rules

You are building and improving the exasol-transformers-extension.

You are using Speq-Skill skills where possible. While using a skill, do not switch to a different task. 
speq:plan is only for planning, speq:implement only for implementing, speq:record only for recoding the results.
Use Speq-skills Codex default models whenever possible for these skills.

Do not make assumptions. Ask the developer instead.

### Directory Structure

```
.exasol_transformers_extension/
├── deployment/       # Deployment related functionality and constant configuration parameters.
├── resources/        # UDF SQL Templates.
├── udfs/callers/     # Python Callers for UDFs.
├── udfs/models/      # UDF Implementaions. Pediction udfs just contain a transfomation pipeline configuration.
├── udfs/models/prediction_tasks/   # Prediction_Task implementations.
├── udfs/models/transformations/    # Transformation implementations for transforming the batch_df.
├── utils/            # untilities related to model storage/loading, device management and dataframe operations.
├── deploy.py         # cli call for deploying the Extension
├── install_default_models.py       # cli call for installling all default models in the bfs
└── upload_model.py                 # cli call for umploading a local model to the bfs

.test/
├── fixtures/         # All Fixtures for test setup go here
├── integration_tests/utils         # utilities for integration tests
├── integration_tests/with_db       # tests for Transformers-Extension/Exasol Database integration
├── integration_tests/without_db    # tests for transformers/Transformers-Extension integration
├── unit/deployment/
├── unit/transformations/
├── unit/udf_wrapper_params/        # all parameterization files for prediction udf unit tests
├── unit/udfs/
├── unit/utils/                     # utilities for unti tests
├── utils                           # utilities for all tests
└── recorder-agent.md            # deterministic spec merge
```

## Testing Rules

### Unit Tests

All parameterization classes for prediction udf unit tests SHALL be placed in unit/udf_wrapper_params/<udf_name>/


### Integration Tests

Only run integration tests if you have the necessary permission. Otherwise, the developer needs to run them.
with_db integration tests need an external docker-db to be setup with "poetry run nox -s "start_database", 
and the tests then need to be configured to use this database.

Do not run the entire Integration test suite. Only run necessary tests.

Integration tests SHALL use existing test fixtures wherever possible.

## Mission Reference for speq CLI

See `specs/mission.md` for purpose, tech stack, commands, and architecture of the speq CLI.
