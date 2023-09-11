# Transformers Extension 0.5.0, released 2023-09-11

Code name: Support for transformer 4.31


## Summary

This release makes the extension compatible with Huggingface transformers v.4.31.0 and their new model cache format. 
Furthermore, it makes the deployment scripts compatible with Exasol v8 by enabling encryption 
and allows the user to configure the TLS verification.

### Features

 - #88: Added custom matcher functions for unit tests
 - #103: Added option to toggle use of TLS certificate validation for Database connection
 - #42: Update transformers to 4.31 and adapt the model uploader

### Bug Fixes

 - #89: Fixed the content of error code config file
 - #100: Enabled encryption for all pyexasol connection to be compatible with Exasol 8
 - #84: Reactivated test after move to AWS
 - #128: Fix release workflow and remove splitting the SLC

### Refactorings

 - #24: Added model counters to unit tests of prediction UDFs
 - #95: Removed setup.py
 - #107: Use SLCT api for building the language container
 - #108: Use itde pytest plugin for tests
 - #110: Splitted SLC into dependency and release build step
 - #8: Moved CI-tests to AWS
 - #115: Refactored ModelDownloaderUDF
 - #121: Use matchers in without db integration tests

### Documentation

 - #93: Added the Developer Guide
 - #126: Add documentation for token for private models in prediction UDFs
  

    
  
