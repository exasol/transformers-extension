# Transformers Extension 0.4.0, released 2023-03-31

Code name: Added Zero-Shot model and error handling structure


## Summary

This release introduces a new UDF script for Zero-Shot text classification. 
Moreover, this version enables users to use custom models located in local 
filesystem or private repositories. In addition, this release includes an error 
handling mechanism to handle errors that may occur during model loading or 
one of the prediction stages.

### Features

 - #11: Converted DownloadUDF to SET UDF 
 - #58: Added setup to upload models from local filesystem
 - #47: Added rank column to model results returning top-k predictions
 - #72: Added authentication token to download private models
 - #64: Added Zero-Shot test classification
 - #25: Added error handling structure

### Documentation

 - #87: Updated User Guide with error_message column 
  

    
  
