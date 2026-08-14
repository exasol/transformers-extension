-- this script is created automatically. Call 'write_create_script' if you need to update it.

CREATE OR REPLACE PYTHON3_TE SET SCRIPT "TE_MODEL_DOWNLOADER_UDF"(
    bucketfs_conn VARCHAR(2000000),
    sub_dir VARCHAR(2000000),
    model_name VARCHAR(2000000),
    task_type VARCHAR(2000000),
    token_conn VARCHAR(2000000)
) EMITS (
    model_path_in_udfs VARCHAR(2000000),
    model_path_of_tar_file_in_bucketfs VARCHAR(2000000)
) AS

"""
Caller for ModelDownloaderUDF
"""

from exasol_transformers_extension.udfs.models.model_downloader_udf import (
    ModelDownloaderUDF,
)

udf = ModelDownloaderUDF(exa)


def run(ctx):
    """
    run function for ModelDownloaderUDF
    """
    return udf.run(ctx)


/
-- next call:

CREATE OR REPLACE PYTHON3_TE SCALAR SCRIPT "INSTALL_AI_DEFAULT_MODEL_UDF"(...)
       EMITS (
    model_path_in_udfs VARCHAR(2000000),
    model_path_of_tar_file_in_bucketfs VARCHAR(2000000)
) AS

"""
Caller for InstallDefaultModelsUDF
"""

from exasol_transformers_extension.udfs.models.install_default_models_udf import (
    InstallDefaultModelsUDF,
)

udf = InstallDefaultModelsUDF(exa)


def run(ctx):
    """
    run function for InstallDefaultModelsUDF
    """
    return udf.run(ctx)


/
-- next call:

CREATE OR REPLACE PYTHON3_TE SCALAR SCRIPT "TE_LIST_MODELS_UDF"(
    bucketfs_conn VARCHAR(2000000),
    sub_dir VARCHAR(2000000)
) EMITS (
    bucketfs_conn VARCHAR(2000000),
    sub_dir VARCHAR(2000000),
    model_name VARCHAR(2000000),
    task_type VARCHAR(2000000),
    model_path VARCHAR(2000000),
    error_message VARCHAR(2000000) ) AS

"""
Caller for ListModelsUDF
"""

from exasol_transformers_extension.udfs.models.ls_models_udf import (
    ListModelsUDF,
)

udf = ListModelsUDF(exa)


def run(ctx):
    """
    run function for ListModelsUDF
    """
    return udf.run(ctx)


/
-- next call:

CREATE OR REPLACE PYTHON3_TE SET SCRIPT "AI_CUSTOM_CLASSIFY_EXTENDED"(
    device_id INTEGER,
    bucketfs_conn VARCHAR(2000000),
    sub_dir VARCHAR(2000000),
    model_name VARCHAR(2000000),
    text_data VARCHAR(2000000),
    return_ranks VARCHAR(2000000)
    ORDER BY model_name ASC,bucketfs_conn ASC,sub_dir ASC
)EMITS (
    bucketfs_conn VARCHAR(2000000),
    sub_dir VARCHAR(2000000),
    model_name VARCHAR(2000000),
    text_data VARCHAR(2000000),
    return_ranks VARCHAR(2000000),
    label VARCHAR(2000000),
    score DOUBLE,
    rank INTEGER,
    error_message VARCHAR(2000000) ) AS

"""
Caller for AiCustomClassifyUDF
"""

from exasol_transformers_extension.udfs.models.ai_custom_classify_extended_udf import (
    AiCustomClassifyUDF,
)

udf = AiCustomClassifyUDF(exa)


def run(ctx):
    """
    run function for AiCustomClassifyUDF
    """
    return udf.run(ctx)


/
-- next call:

CREATE OR REPLACE PYTHON3_TE SET SCRIPT "AI_ENTAILMENT_EXTENDED"(
    device_id INTEGER,
    bucketfs_conn VARCHAR(2000000),
    sub_dir VARCHAR(2000000),
    model_name VARCHAR(2000000),
    first_text VARCHAR(2000000),
    second_text VARCHAR(2000000),
    return_ranks VARCHAR(2000000)
    ORDER BY model_name ASC,bucketfs_conn ASC,sub_dir ASC
)EMITS (
    bucketfs_conn VARCHAR(2000000),
    sub_dir VARCHAR(2000000),
    model_name VARCHAR(2000000),
    first_text VARCHAR(2000000),
    second_text VARCHAR(2000000),
    return_ranks VARCHAR(2000000),
    label VARCHAR(2000000),
    score DOUBLE,
    rank INTEGER,
    error_message VARCHAR(2000000) ) AS

"""
Caller for AiEntailmentExtendedUDF
"""

from exasol_transformers_extension.udfs.models.ai_entailment_extended_udf import (
    AiEntailmentExtendedUDF,
)

udf = AiEntailmentExtendedUDF(exa)


def run(ctx):
    """
    run function for AiEntailmentExtendedUDF
    """
    return udf.run(ctx)


/
-- next call:

CREATE OR REPLACE PYTHON3_TE SET SCRIPT "AI_ANSWER_EXTENDED"(
    device_id INTEGER,
    bucketfs_conn VARCHAR(2000000),
    sub_dir VARCHAR(2000000),
    model_name VARCHAR(2000000),
    question VARCHAR(2000000),
    context_text VARCHAR(2000000),
    top_k INTEGER
    ORDER BY model_name ASC,bucketfs_conn ASC,sub_dir ASC
)EMITS (
    bucketfs_conn VARCHAR(2000000),
    sub_dir VARCHAR(2000000),
    model_name VARCHAR(2000000),
    question VARCHAR(2000000),
    context_text VARCHAR(2000000),
    top_k INTEGER,
    answer VARCHAR(2000000),
    score DOUBLE,
    rank INTEGER,
    error_message VARCHAR(2000000) ) AS

"""
Caller for AiAnswerExtendedUDF
"""

from exasol_transformers_extension.udfs.models.ai_answer_extended_udf import (
    AiAnswerExtendedUDF,
)

udf = AiAnswerExtendedUDF(exa)


def run(ctx):
    """
    run function for AiAnswerExtendedUDF
    """
    return udf.run(ctx)


/
-- next call:

CREATE OR REPLACE PYTHON3_TE SET SCRIPT "AI_FILL_MASK_EXTENDED"(
    device_id INTEGER,
    bucketfs_conn VARCHAR(2000000),
    sub_dir VARCHAR(2000000),
    model_name VARCHAR(2000000),
    text_data VARCHAR(2000000),
    top_k INTEGER
    ORDER BY model_name ASC,bucketfs_conn ASC,sub_dir ASC
)EMITS (
    bucketfs_conn VARCHAR(2000000),
    sub_dir VARCHAR(2000000),
    model_name VARCHAR(2000000),
    text_data VARCHAR(2000000),
    top_k INTEGER,
    filled_text VARCHAR(2000000),
    score DOUBLE,
    rank INTEGER,
    error_message VARCHAR(2000000) ) AS

"""
Caller for AiFillMaskExtendedUDF
"""

from exasol_transformers_extension.udfs.models.ai_fill_mask_extended_udf import (
    AiFillMaskExtendedUDF,
)

udf = AiFillMaskExtendedUDF(exa)


def run(ctx):
    """
    run function for AiFillMaskExtendedUDF
    """
    return udf.run(ctx)


/
-- next call:

CREATE OR REPLACE PYTHON3_TE SET SCRIPT "AI_COMPLETE_EXTENDED"(
    device_id INTEGER,
    bucketfs_conn VARCHAR(2000000),
    sub_dir VARCHAR(2000000),
    model_name VARCHAR(2000000),
    text_data VARCHAR(2000000),
    max_new_tokens INTEGER,
    return_full_text BOOLEAN
    ORDER BY model_name ASC,bucketfs_conn ASC,sub_dir ASC
)EMITS (
    bucketfs_conn VARCHAR(2000000),
    sub_dir VARCHAR(2000000),
    model_name VARCHAR(2000000),
    text_data VARCHAR(2000000),
    max_new_tokens INTEGER,
    return_full_text BOOLEAN,
    generated_text VARCHAR(2000000),
    error_message VARCHAR(2000000)) AS

"""
Caller for AiCompleteExtendedUDF
"""

from exasol_transformers_extension.udfs.models.ai_complete_extended_udf import (
    AiCompleteExtendedUDF,
)

udf = AiCompleteExtendedUDF(exa)


def run(ctx):
    """
    run function for AiCompleteExtendedUDF
    """
    return udf.run(ctx)


/
-- next call:

CREATE OR REPLACE PYTHON3_TE SET SCRIPT "AI_TRANSLATE_EXTENDED"(
    device_id INTEGER,
    bucketfs_conn VARCHAR(2000000),
    sub_dir VARCHAR(2000000),
    model_name VARCHAR(2000000),
    text_data VARCHAR(2000000),
    source_language VARCHAR(2000000),
    target_language VARCHAR(2000000),
    max_new_tokens INTEGER
    ORDER BY model_name ASC,bucketfs_conn ASC,sub_dir ASC
)EMITS (
    bucketfs_conn VARCHAR(2000000),
    sub_dir VARCHAR(2000000),
    model_name VARCHAR(2000000),
    text_data VARCHAR(2000000),
    source_language VARCHAR(2000000),
    target_language VARCHAR(2000000),
    max_new_tokens INTEGER,
    translation_text VARCHAR(2000000),
    error_message VARCHAR(2000000)) AS

"""
Caller for AiTranslateExtendedUDF
"""

from exasol_transformers_extension.udfs.models.ai_translate_extended_udf import (
    AiTranslateExtendedUDF,
)

udf = AiTranslateExtendedUDF(exa)


def run(ctx):
    """
    run function for AiTranslateExtendedUDF
    """
    return udf.run(ctx)


/
-- next call:

CREATE OR REPLACE PYTHON3_TE SET SCRIPT "TE_DELETE_MODEL_UDF"(
    bucketfs_conn VARCHAR(2000000),
    sub_dir VARCHAR(2000000),
    model_name VARCHAR(2000000),
    task_type VARCHAR(2000000)
) EMITS (
    bucketfs_conn VARCHAR(2000000),
    sub_dir VARCHAR(2000000),
    model_name VARCHAR(2000000),
    task_type VARCHAR(2000000),
    success BOOLEAN,
    error_message VARCHAR(2000000)
) AS

"""
Caller for DeleteModelUDF
"""

from exasol_transformers_extension.udfs.models.delete_model_udf import (
    DeleteModelUDF,
)

udf = DeleteModelUDF(exa)


def run(ctx):
    """
    run function for DeleteModelUDF
    """
    return udf.run(ctx)


/
-- next call:

CREATE OR REPLACE PYTHON3_TE SET SCRIPT "AI_SENTIMENT"(
    text_data VARCHAR(2000000)
)EMITS (
    text_data VARCHAR(2000000),
    label VARCHAR(2000000),
    score DOUBLE,
    error_message VARCHAR(2000000) ) AS

"""Caller for AiSentimentUDF"""

from exasol_transformers_extension.udfs.models.ai_sentiment_udf import AiSentimentUDF

udf = AiSentimentUDF(exa)


def run(ctx):
    """run function for AiSentimentUDF"""
    return udf.run(ctx)


/
-- next call:

CREATE OR REPLACE PYTHON3_TE SET SCRIPT "AI_CLASSIFY"(
    text_data VARCHAR(2000000),
    candidate_labels VARCHAR(2000000)
)EMITS (
    text_data VARCHAR(2000000),
    candidate_labels VARCHAR(2000000),
    label VARCHAR(2000000),
    score DOUBLE,
    error_message VARCHAR(2000000) ) AS

"""Caller for AiClassifyUDF"""

from exasol_transformers_extension.udfs.models.ai_classify_udf import AiClassifyUDF

udf = AiClassifyUDF(exa)


def run(ctx):
    """run function for AiClassifyUDF"""
    return udf.run(ctx)


/
-- next call:

CREATE OR REPLACE PYTHON3_TE SET SCRIPT "AI_EXTRACT_ENTITIES"(
    text_data VARCHAR(2000000)
)EMITS (
    text_data VARCHAR(2000000),
    start_pos INTEGER,
    end_pos INTEGER,
    word VARCHAR(2000000),
    entity VARCHAR(2000000),
    score DOUBLE,
    error_message VARCHAR(2000000) ) AS

"""Caller for AiExtractEntitiesUDF"""

from exasol_transformers_extension.udfs.models.ai_extract_entities_udf import (
    AiExtractEntitiesUDF,
)

udf = AiExtractEntitiesUDF(exa)


def run(ctx):
    """run function for AiExtractEntitiesUDF"""
    return udf.run(ctx)


/
-- next call:

CREATE OR REPLACE PYTHON3_TE SET SCRIPT "AI_EXTRACT_EXTENDED"(
    device_id INTEGER,
    bucketfs_conn VARCHAR(2000000),
    sub_dir VARCHAR(2000000),
    model_name VARCHAR(2000000),
    text_data VARCHAR(2000000),
    aggregation_strategy VARCHAR(2000000)
    ORDER BY model_name ASC,bucketfs_conn ASC,sub_dir ASC
)EMITS (
    bucketfs_conn VARCHAR(2000000),
    sub_dir VARCHAR(2000000),
    model_name VARCHAR(2000000),
    text_data VARCHAR(2000000),
    aggregation_strategy VARCHAR(2000000),
    start_pos INTEGER,
    end_pos INTEGER,
    word VARCHAR(2000000),
    entity VARCHAR(2000000),
    score DOUBLE,
    error_message VARCHAR(2000000) ) AS

"""
Caller for AiExtractExtendedUDF
"""

from exasol_transformers_extension.udfs.models.ai_extract_extended_udf import (
    AiExtractExtendedUDF,
)

udf = AiExtractExtendedUDF(exa)


def run(ctx):
    """
    run function for AiExtractExtendedUDF
    """
    return udf.run(ctx)


/
-- next call:

CREATE OR REPLACE PYTHON3_TE SET SCRIPT "AI_CLASSIFY_EXTENDED"(
    device_id INTEGER,
    bucketfs_conn VARCHAR(2000000),
    sub_dir VARCHAR(2000000),
    model_name VARCHAR(2000000),
    text_data VARCHAR(2000000),
    candidate_labels VARCHAR(2000000),
    return_ranks VARCHAR(2000000)
    ORDER BY model_name ASC,bucketfs_conn ASC,sub_dir ASC
)EMITS (
    bucketfs_conn VARCHAR(2000000),
    sub_dir VARCHAR(2000000),
    model_name VARCHAR(2000000),
    text_data VARCHAR(2000000),
    candidate_labels VARCHAR(2000000),
    return_ranks VARCHAR(2000000),
    label VARCHAR(2000000),
    score DOUBLE,
    rank INTEGER,
    error_message VARCHAR(2000000) ) AS

"""
Caller for AiClassifyExtendedUDF
"""

from exasol_transformers_extension.udfs.models.ai_classify_extended_udf import (
    AiClassifyExtendedUDF,
)

udf = AiClassifyExtendedUDF(exa)


def run(ctx):
    """
    run function for AiClassifyExtendedUDF
    """
    return udf.run(ctx)


/
-- next call:

