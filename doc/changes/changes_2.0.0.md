# Transformers Extension 2.0.0, t.b.d

Code name: 

## Summary


### Features

- #243: Added an option to deploy scripts in a SaaS database. 

### Bugs

- #237: Fixed reference to python-extension-common

### Documentation

- #210: Fixed typos in user guide.
- #247: Updated documentation including the deployment options in SaaS.

### Refactorings

- #216: Simplified model path constructions, consolidating them into one function
- #228: Now use python-extension-common for the language container deployment.
- #232: Added Class which holds model information  
- #217: Refactored PredictionUDFs and LoadLocalModel so that LoadLocalModel constructs the bucketfs model file path
- #230: Updated supported python version to >= Python 3.10
- #236: Moved to the PathLike bucketfs interface.

### Security 
