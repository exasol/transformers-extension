# Transformers Extension 3.0.0, 2025-11-06

Code name: Improved ranking

## Summary

In this release, we added a `return_ranks` and a `rank` column to some UDFs, 
to improve the flexibility in building Pipelines using the Exasol Transformers Extension.

#### BREAKING CHANGES:

* The Sequence-Classification UDF's (single text, text pair) now require a `return_ranks` input Column, which is used 
  to determine how many results per input should be returned.
* The Sequence-Classification UDF's (single text, text pair) now return a `rank` column with the rank of the results.
* The Zero-Shot-Classification UDF now require a `return_ranks` input Column, which is used to determine how many 
  results per input should be returned.

Please refer to the [user guide](../user_guide/user_guide.md) for Details.

## Features


* #329: Added return_ranks to Zero-Shot-Classification UDF
* #326: Added parameters `rank` and `return_ranks` to sequence classification single text udf
* #327: Added parameters `rank` and `return_ranks` to sequence classification text pair udf

## Refactorings

* #311: Updated transitive dependencies
* #337: Moved CI Integration tests from AWS CodeBuild to GitHub Actions

## Security

* #333: Resolved CVE-2025-3730 for torch by bumping version to ^2.8.0 and setting requires-python = ">=3.10.0,<3.15"

## Internal

* #333: Resolved CVE-2025-8869 for transitive dependency pip by re-locking dependencies
