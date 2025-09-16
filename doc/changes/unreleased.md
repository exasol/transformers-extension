# Transformers Extension X.X.X, T.B.D

T.B.D

## Summary

T.B.D

## Features

* #149: Added Python API for uploading a model to a given BucketFS location

## Bugfixes

* #285: Fixed logic in model output quality checks of UDF integration test
* #285: Deactivated SaaS tests because of a broken API

## Documentation

* #319: Updated TE User Guide

## Refactorings

* Updated  tornado (6.4.2 -> 6.5.1)
* #201: Added python toolbox to project
* #294: Improved linter score
* #305: Updated python toolbox to 1.1.0 & added basic typing checks
* #295: Activated type check in .pre-commit-config.yaml
* #296: Activated code formatting in .pre-commit-config.yaml
* #311: Updated transitive dependencies urllib3 (2.4.0 -> 2.5.0) & requests (2.32.3 -> 2.32.4)
* #317: Added function `download_transformers_model()`
