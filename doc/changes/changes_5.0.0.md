# Transformers Extension 5.0.0, 2026-07-28

Code name: Dependency Updates

## Summary

In this Release, the transformers package was updated from version 4 to version 5.

### BREAKING CHANGES:

In transformers version 5, the pipeline for question-answering has
been removed. For our AI_ANSWER_EXTEDNED UDF, we replaced it with 
the text-generation pipeline. You will not be able to use you old question-answering
models with this new implementation, switch to text-generation models instead.
(The translation pipeline was also removed, but we decided to maintain it ourselves for 
now, so the AI_TRANSLATE_EXTENDED UDF is unaffected.)


## Security

* Updated gitpython (3.1.52 -> 3.1.57)
* Updated exasol-python-extension-common (0.15.0 -> 0.16.0)

## Refactorings

* #395: Update to exasol-toolbox 10.0.0
* #402: Update to exasol-toolbox 10.2.1
* #407: Updated to transformers version 5

