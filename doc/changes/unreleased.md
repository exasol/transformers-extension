# Transformers Extension X.X.X, T.B.D

T.B.D

## Summary

T.B.D

## Features

* #329: Added return_ranks to Zero-Shot-Classification UDF
* #328: Added parameters `rank` and `return_ranks` to sequence classification udf's

## Bugfixes

## Documentation

## Refactoring

* #311: Updated transitive dependencies
* #333: Updated dependency declaration to `pyexasol`

## Security

* #333: Resolved CVE-2025-3730 for torch by bumping version to ^2.8.0 and setting requires-python = ">=3.10.0,<3.15"

## Internal

* #333: Resolved CVE-2025-8869 for transitive dependency pip by re-locking dependencies