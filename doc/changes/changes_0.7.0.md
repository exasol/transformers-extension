# Transformers Extension 0.7.0, released 2023-12-19

Code name: Split SLC actions


## Summary

In this Release split the container uploading and registration into two separate actions. Additionally, 
a workflow for checking the correctness of the version number in multiple places was added. Apart from that there 
are some refactorings for better usability and the  Cryptography dependency version has been upgraded to 41.0.7

### Features

  - #151: Made the container uploading and language registration two separate actions
  - #167: Added version check workflow

### Refactorings

  - #144: Extracted base_model_udf.load_models into separate class
  - #159: Refactored LanguageContainer Deployer to make it reusable by other extensions

### Security

  - #144: Updated Cryptography to version 41.0.7
  