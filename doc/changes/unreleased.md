# Transformers Extension X.X.X, T.B.D

Code name: T.B.D

## Summary

T.B.D

### BREAKING CHANGES:


## Features

 * #351: Added functionality for installing default models.
 * #378. Added creation of default BucketFS-Connection to deploy command
 * #383: Added Transformation for adding columns to DataFrame and filling them with default values.
 * #381: Added Transformation for removing columns from DataFrame
 * #353: Added "AI_SENTIMENT" UDF
 * #390: Added "AI_CLASSIFY" and "AI_EXTRACT_ENTITIES" UDF's
 * #391: Added sql create_script to create all UDF's in Database.
## Security

## Bugfixes

* Avoid storing model archives twice during BucketFS uploads.

## Documentation

## Refactorings

* Updated SaaS CI to run the selected `with_db` integration tests in a single job.
* Split on-prem slow integration tests into separate per-file jobs and correctly pass
  paths containing special characters.
* #400: Replaced project `tarfile` usage with `fastar` for BucketFS model archive creation and extraction tests.
* #400: Use uncompressed `.tar` archives for new model uploads while retaining support for existing `.tar.gz` archives.
