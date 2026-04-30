# Transformers Extension X.X.X, T.B.D

Code name: T.B.D

## Summary

### BREAKING CHANGES:

* The `max_length` parameter has been renamed to `max_new_tokens`, and its behavior changed. Both of these changes where done in accordance with changes in the [transformers library](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.TextGenerationPipeline).
* All prediction UDFs have been renamed:

| Old UDF Name                               | New UDF Name                |
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

 * #351: Added functionality for installing default models.
 * #378. Added creation of default BucketFS-Connection to deploy command
 * #383: Added Transformation for adding columns to DataFrame and filling them with default values.
 * #381: Added Transformation for removing columns from DataFrame
 * #353: Added "AI_SENTIMENT" UDF

## Security

* Updated exasol-integration-test-docker-environment (4.4.1 -> 5.0.0)
* Updated exasol-script-languages-container-tool (3.4.1 -> 3.5.0)
* Updated exasol-saas-api (2.3.0 -> 2.6.0)
* #376: Fixed vulnerabilities by updating dependencies

This release fixes vulnerabilities by updating dependencies:

| Dependency   | Affected | Vulnerabilities     | Fixed in | Updated to |
|--------------|----------|---------------------|----------|------------|
| black        | 25.9.0   | GHSA-3936-cmfr-pm3m | 26.3.1   | 25.12.0    |
| cryptography | 46.0.1   | GHSA-r6ph-v2qm-q3c2 | 46.0.5   | 46.0.6     |
| cryptography | 46.0.1   | GHSA-m959-cc7f-wv43 | 46.0.6   | 46.0.6     |
| filelock     | 3.19.1   | GHSA-w853-jp5j-5j7f | 3.20.1   | 3.25.2     |
| filelock     | 3.19.1   | GHSA-qmgc-5h2g-mvrw | 3.20.3   | 3.25.2     |
| pip          | 25.3     | GHSA-6vgw-5pg2-w6jp | 26.0     | 26.0.1     |
| pyasn1       | 0.6.1    | GHSA-63vm-454h-vhhq | 0.6.2    | 0.6.3      |
| pyasn1       | 0.6.1    | GHSA-jr27-m4p2-rc6r | 0.6.3    | 0.6.3      |
| pygments     | 2.19.2   | GHSA-5239-wwwm-4pmq | 2.20.0   | 2.20.0     |
| pynacl       | 1.6.0    | GHSA-mrfv-m5wm-5w6w | 1.6.2    | 1.6.2      |
| pyopenssl    | 25.1.0   | GHSA-vp96-hxj8-p424 | 26.0.0   | (removed)  |
| pyopenssl    | 25.1.0   | GHSA-5pwr-322w-8jr4 | 26.0.0   | (removed)  |
| requests     | 2.32.5   | GHSA-gc5v-m9x4-r6x2 | 2.33.0   | 2.33.1     |
| tornado      | 6.5.2    | GHSA-78cv-mqj4-43f7 | 6.5.5    | 6.5.5      |
| tornado      | 6.5.2    | GHSA-qjxf-f2mg-c6mc | 6.5.5    | 6.5.5      |
| urllib3      | 2.5.0    | GHSA-gm62-xv2j-4w53 | 2.6.0    | 2.6.3      |
| urllib3      | 2.5.0    | GHSA-2xpw-w6gg-jr37 | 2.6.0    | 2.6.3      |
| urllib3      | 2.5.0    | GHSA-38jv-5279-wg99 | 2.6.3    | 2.6.3      |
| virtualenv   | 20.34.0  | GHSA-597g-3phw-6986 | 20.36.1  | 21.2.0     |

## Bugfixes

* #343: Fixed `max_length` parameter being ignored, renamed `max_length` to `max_new_tokens`

## Documentation

* #204: Split the user_guide into multiple files
* #253, #341, #342: Fixed links and improved parameter description in user guide

## Refactorings

* #346: Changed translation_udf unit tests to use StandaloneUdfMock
* #323: Standardized udf parameter order (changes in TE_DELETE_MODEL_UDF, TE_MODEL_DOWNLOADER_UDF)
* #350: Renamed all prediction UDFs.
* #358: Refactored deployment configuration
* #348: Improved mock model-output for testing max_new_tokens handling in translation unit tests
* #360: Updated to exasol-toolbox 5.1.1 and relocked vulnerable transitive dependencies
* #370: Pulled new class PredictionTask out of BaseModelUdf
* #372: Added Transformation Protocol and extracted GetPredictionFromBatch into Transformations
* #374: Extracted Span handling into Transformations
* #375: Added implementation for a generalized extract_unique_param_based_dataframes function
