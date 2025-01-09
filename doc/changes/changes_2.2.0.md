# Transformers Extension 2.2.0, T.B.D

Code name: T.B.D

## Summary

T.B.D

### Features

- #260: Added span input and output types for zero-shot-classification-udf

### Bugs

- #272: Fixed unit tests assertions not working correctly
- #275: Fixed a bug where models returning unexpected results was not handled correctly

### Documentation

n/a

### Refactorings

- #273: Refactored unit tests for token_classification_udf to use StandAloneUDFMock, made params files more maintainable
- #271: Moved test cases which only pertain to the base udf to base udf unit tests
- #274: Refactored unit tests for zero_shot_text_classification_udf to use StandAloneUDFMock, made params files more maintainable

### Security

n/a