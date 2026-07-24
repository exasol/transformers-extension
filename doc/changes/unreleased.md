# Transformers Extension X.X.X, T.B.D

Code name: T.B.D

## Summary

T.B.D

### BREAKING CHANGES:

In transformers version 5, the pipeline for question-answering has
been removed. For our AI_ANSWER_EXTEDNED UDF, we replaced it with 
the text-generation pipeline. You will not be able to use you old question-answering
models with this new implementation, switch to text-generation models instead.
(The translation pipeline was also removed, but we decided to maintain it ourselves for 
now, so the AI_TRANSLATE_EXTENDED UDF is unaffected.)

## Features

## Security

## Bugfixes

## Documentation

## Refactorings

* #395: Update to exasol-toolbox 10.0.0
* #402: Update to exasol-toolbox 10.2.1
* #407: Updated to transformers version 5

