# Transformers Extension X.X.X, T.B.D

Code name: T.B.D

## Summary

T.B.D

### BREAKING CHANGES:


## Features

## Security

## Bugfixes

* Avoid storing model archives twice during BucketFS uploads.

## Documentation

## Refactorings

* Updated SaaS CI to run the selected `with_db` integration tests in a single job.
* Split on-prem slow integration tests into separate per-file jobs and correctly pass
  paths containing special characters.
* Replaced project `tarfile` usage with `fastar` for BucketFS model archive creation and extraction tests.
