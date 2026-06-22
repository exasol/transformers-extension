# Mission: Exasol Transformers Extension

> Exasol Transformers Extension lets Exasol users run NLP inference inside an Exasol database by deploying Python UDFs that use Hugging Face Transformers models stored in BucketFS.

## Problem Statement

Exasol users who want to apply machine learning to their database data need a way to run supported NLP inference tasks directly in Exasol. The extension focuses on running prediction UDFs locally inside Exasol Database, using models installed in BucketFS before prediction. Running models where the data already lives removes the need for expensive data transfer to external inference services and helps ensure user data is not shared with third parties.

## Target Users

| Persona | Goal | Key Workflow |
|---------|------|--------------|
| Exasol users interested in machine learning on their data | Run NLP predictions against data in Exasol using supported pretrained Hugging Face Transformers models | Deploy the Transformers Extension on an Exasol database, create BucketFS connections if needed, choose a prediction UDF, install default or custom models, call the UDF on database data with the correct model |

## Core Capabilities

1. **Run NLP prediction UDFs** - Execute supported NLP inference tasks inside Exasol Database and return prediction results through SQL query output.
2. **Provide default-model prediction UDFs** - Offer prediction UDFs that use default models for common tasks with minimal user configuration.
3. **Provide customizable prediction UDFs** - Offer prediction UDFs where users can select compatible models and task parameters for their use case.
4. **Manage models in BucketFS** - Support installing, listing, and deleting Hugging Face models in Exasol BucketFS as supporting functionality for prediction.
5. **Deploy extension assets** - Deploy the Script Language Container, UDF scripts, and required connection objects needed to run the extension in Exasol.

## Out of Scope

- Windows and macOS support.
- Model training and fine-tuning.
- Non-NLP tasks.
- Models that do not support at least one supported task type.
- Managing Exasol clusters.
- Online inference outside Exasol Database.
- Sharing user data with anything other than the selected model, data preprocessing transformations, and UDF output.
- Other non-goals not listed here may still be out of scope when they conflict with the mission of running NLP inference inside Exasol.

## Domain Glossary

| Term | Definition |
|------|------------|
| Exasol Database | The database runtime where the extension deploys UDF scripts and executes NLP inference against user data. |
| Transformers library | The Hugging Face Python library used by the extension to load pretrained models and run supported NLP prediction tasks. |
| Hugging Face model | A pretrained model, and its tokenizer where applicable, selected for one of the supported Transformers task types. |
| UDF | User Defined Function; an Exasol script callable from SQL that executes extension logic inside the database. |
| Prediction UDF | A UDF that loads an installed model and runs NLP inference for a supported task. |
| BucketFS | Exasol's file storage used by the extension to store Hugging Face models and extension assets for database-side execution. |
| Script Language Container | The Exasol runtime package that provides Python and required dependencies for UDF execution. |
| Task type | The Transformers task category a model supports, such as text classification, question answering, fill-mask, text generation, token classification, translation, or zero-shot classification. |

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Language | Python 3.10 through 3.14 | Main implementation language for deployment code, model management, UDF logic, and tests. |
| Packaging | Poetry | Dependency management, installation, and package builds. |
| Runtime libraries | PyTorch, Hugging Face Transformers, pandas | Model inference, Transformers task execution, and tabular input/output processing. |
| CLI | Click | Command-line entry points for deployment and model installation workflows. |
| Exasol integration | pyexasol, exasol-bucketfs, exasol-python-extension-common | Database connectivity, BucketFS access, and shared extension deployment support. |
| Testing | pytest, nox, pytest-exasol-slc, pytest-exasol-extension, exasol-udf-mock-python | Unit tests, integration tests, UDF tests, and test orchestration. |
| Formatting and linting | black, isort, pylint, ruff, mypy | Code style, linting, and type-checking support configured by the project. |

## Commands

```bash
# Install released package
pip install exasol-transformers-extension

# Install development environment
poetry install

# Build package
poetry build

# Deploy extension to an Exasol database
python -m exasol_transformers_extension.deploy <options>

# Upload a custom model to BucketFS
python -m exasol_transformers_extension.upload_model <options>

# Install default models to BucketFS
python -m exasol_transformers_extension.install_default_models <options>

# Run unit tests
poetry run -- nox -s test:unit

# Start test database for integration tests
poetry run -- nox -s start_database

# Run integration test groups
poetry run -- nox -s onprem_integration_tests
poetry run -- nox -s saas_integration_tests
poetry run -- nox -s without_db_integration_tests

# Format code
poetry run -- nox -s format:fix

# Check formatting
poetry run -- nox -s format:check
```

Command availability was verified in the current workspace with Poetry using Python 3.12: `poetry install`, `poetry build`, Nox session discovery, and CLI help for deployment and model installation commands succeeded. Full deployment, model upload, model installation, unit tests, and integration tests were not executed as part of mission creation; they require external systems, model downloads, or longer-running validation.

## Project Structure

```text
transformers-extension/
├── exasol_transformers_extension/        # Python package for deployment, UDFs, resources, and model utilities
│   ├── deployment/                       # Script Language Container and UDF deployment support
│   ├── resources/templates/              # SQL/Jinja templates for deployed UDF scripts
│   ├── udfs/callers/                     # Exasol UDF script entry points
│   ├── udfs/models/                      # UDF implementations, transformations, and prediction tasks
│   └── utils/                            # BucketFS, model loading, model transfer, and device utilities
├── doc/                                  # User guide, developer guide, design notes, dependencies, and changelog
├── test/                                 # Unit tests, integration tests, fixtures, and test utilities
├── scripts/                              # Project helper scripts
├── noxfile.py                            # Test, formatting, integration, SLC export, and database sessions
├── noxconfig.py                          # Nox and toolbox project configuration
├── pyproject.toml                        # Package metadata, dependencies, build system, and tool configuration
└── version.py                            # Project version file used by release tooling
```

## Architecture

Exasol Transformers Extension is an Exasol extension packaged as a Python library, deployment CLI, BucketFS model-management utilities, and Python UDF scripts. Models are stored in BucketFS, loaded by prediction UDFs, and inference results are emitted back into SQL query results.

The high-level flow is:

1. A user installs the Python package or builds it from source.
2. A user deploys the extension to Exasol, including the Script Language Container, UDF scripts, and connection objects as needed.
3. A user installs default or custom Hugging Face models into BucketFS.
4. A user calls a prediction UDF from SQL with the correct task type, model, and input parameters.
5. The UDF loads the model locally in Exasol Database, applies preprocessing transformations, runs prediction through Hugging Face Transformers, and emits output rows.

The implementation separates deployment, model management, UDF callers, UDF model logic, prediction task logic, and dataframe transformations into dedicated package areas.

## Constraints

- **Technical**: The extension is not tested on Windows or macOS. Offered UDFs must run in Exasol DB 7.1 or later. Models must run locally in Exasol Database. Models must be downloaded prior to prediction if they are not yet installed. Prediction UDFs require models compatible with at least one supported task type.
- **Business**: Private-model tokens, Exasol passwords, and BucketFS passwords must be handled as secrets.
- **Data handling**: User data must not be shared with anything other than the selected model, data preprocessing transformations, and UDF output.
- **Performance**: Large pretrained models should not be downloaded for every prediction task; installed models should be reused from BucketFS.

## External Dependencies

| Service | Purpose | Failure Impact |
|---------|---------|----------------|
| Exasol Database | Hosts deployed UDF scripts and runs NLP inference on database data | Prediction UDFs cannot execute. |
| Exasol BucketFS | Stores installed Hugging Face models, Script Language Container assets, and files needed by UDF execution | Models and extension assets cannot be installed, loaded, listed, or deleted. |
| Hugging Face Hub | Source for downloading default or custom pretrained models | New model installation from Hugging Face fails; already installed models can still be used. |
| GitHub releases | Source for prebuilt Script Language Container artifacts | Deployment that relies on downloading prebuilt containers can fail. |
| PyPI/package indexes | Source for installing the Python package and dependencies | New package installation or environment setup can fail. |
| Exasol SaaS APIs | BucketFS access path for SaaS deployments | SaaS model and asset upload workflows can fail. |
