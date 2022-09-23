# Transformers Extension 0.1.0, released 2022-09-23

Code name: Add a downloader UDF and a set of prediction UDFs using the transformers API


## Summary
This is the initial release of the transformers-extension which provides a 
downloader UDF that allows us to store the pre-trained machine learning model 
by transformers and a set of Prediction UDFs that allow the downloaded model 
to be cached and used through the transformers API.

This version provides the following machine learning tasks:

* Sequence Classification
* Question Answering, 
* Filling Mask
* Text Generation
* Token Classification
* Text Translation.

### Features

  - #1: Added the initial setup of the project
  - #5: Prepared the skeleton of the project
  - #4: Added model downloader UDF 
  - #9: Created sequence classification UDF for single text
  - #14: Created sequence classification UDF for pair text
  - #16: Created question answering UDF
  - #21: Added parameter specifying GPU device
  - #26: Created masked language modelling UDF
  - #29: Created text generation UDF
  - #31: Created token classification UDF
  - #28: Added top_k result returning feature to question answering UDF
  - #33: Added text translation UDF
  - #48: Prepared the initial release
  
### Bug Fixes

  - #2: Renamed master branch to main
  - #18: Corrected model filtering in prediction UDFs
  - #50: Fixed release_droid configuration
  - #52: Reduced disk space used by the machine during releasing

### Refactoring

 - #12: Updated method for generating bucket udf path
 - #35: Setup masked language modelling pipeline once
 - #20: Applied same API call across all prediction UDFs
 - #19: Inherent prediction UDF classes from same base class

### Documentation

 - #44: Added User Guide

  
