# Transformers Extension 2.2.0, 2025-01-21

Code name: Bugfix for token classification

## Summary

This release includes a bugfix for handling unexpected results in the token classification udf, 
as well as internal refactorings for the unit tests.

### Bugs

- #272: Fixed unit tests assertions not working correctly
- #275: Fixed a bug where models returning unexpected results was not handled correctly

### Refactorings

- #273: Refactored unit tests for token_classification_udf to use StandAloneUDFMock, made params files more maintainable
- #271: Moved test cases which only pertain to the base udf to base udf unit tests
