# Transformers Extension 5.2.0, 2026-09-02

Code name: Improved model upload

## Summary

In this Release, the upload of models to the BucketFS has 
gotten a rework to allow the use of bigger models.

We also added a new script which installs all available UDF's in Database.

## Features

 * #391: Added sql create_script to create all UDF's in Database.

## Bugfixes

* Avoid storing model archives twice during BucketFS uploads.
* Fixed SLC name in TeLanguageContainerDeployer to match new SLC name

## Refactorings

* Updated SaaS CI to run the selected `with_db` integration tests in a single job.
* Split on-prem slow integration tests into separate per-file jobs and correctly pass
  paths containing special characters.
* #400: Replaced project `tarfile` usage with `fastar` for BucketFS model archive creation and extraction tests.
* #400: Use uncompressed `.tar` archives for new model uploads while retaining support for existing `.tar.gz` archives.
