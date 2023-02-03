# Transformers Extension 0.3.0, released 2023-02-03

Code name: Simplify language container setup and fix torch version 1.11.0


## Summary

This version allows users to install the language container in one step. 
The version of language container to be installed is given to the installation 
script, therefore there is no need to download the container file separately. 
Moreover, this release fixed torch version to 1.11.0, so that we avoided to 
package unused nvidia dependencies existing later versions of torch.

### Features

 n/a
  
### Bug Fixes

 n/a

### Refactoring

 - #76: Updated torch version to 1.11.0
 - #79: Deployed SLC in on step

### Documentation

 n/a
    
  
