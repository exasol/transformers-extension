# Transformers Extension X.X.X, T.B.D

Code name: T.B.D

## Summary

T.B.D

#### BREAKING CHANGES:

* The `max_length` parameter has been renamed to `max_new_tokens`, and its behavior changed. 
Both of these changes where done in accordance with changes in the [transformers library](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.TextGenerationPipeline).
* All prediction UDFs have been renamed:

| Old UDF name                               | new UDF name                |
|--------------------------------------------|-----------------------------|
| TE_FILLING_MASK_UDF                        | AI_FILL_MASK_EXTENDED       |
| TE_QUESTION_ANSWERING_UDF                  | AI_ANSWER_EXTENDED          |
| TE_SEQUENCE_CLASSIFICATION_SINGLE_TEXT_UDF | AI_CUSTOM_CLASSIFY_EXTENDED |
| TE_SEQUENCE_CLASSIFICATION_TEXT_PAIR_UDF   | AI_ENTAILMENT_EXTENDED      |
| TE_TEXT_GENERATION_UDF                     | AI_COMPLETE_EXTENDED        |
| TE_TRANSLATION_UDF                         | AI_TRANSLATE_EXTENDED       |
| TE_TOKEN_CLASSIFICATION_UDF                | AI_EXTRACT_EXTENDED         |
| TE_ZERO_SHOT_CLASSIFICATION_UDF            | AI_CLASSIFY_EXTENDED        |



## Features

## Bugfixes

 * #343: Fixed max_length parameter being ignored, renamed max_length to max_new_tokens

## Documentation

 * #204: Split the user_guide into multiple files
 * #253, #341, #342: Link fixes and better parameter description in user guide

## Refactoring

 * #346: Changed translation_udf unit tests to use StandaloneUdfMock
 * #323: Standardized udf parameter order (changes in TE_DELETE_MODEL_UDF, TE_MODEL_DOWNLOADER_UDF)
 * #350: Renamed all prediction UDFs.
 * #360: Updated to exasol-toolbox 5.1.1 and relocked vulnerable transitive dependencies

## Security

 Updated urllib3 (2.5.0 -> 2.6.1)

## Internal

 Updated urllib3 (2.5.0 -> 2.6.2)
 Updated exasol-integration-test-docker-environment (4.4.1 -> 5.0.0)
 Updated exasol-script-languages-container-tool (3.4.1 -> 3.5.0)
 Updated exasol-saas-api (2.3.0 -> 2.6.0)

