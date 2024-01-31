# Transformers Extension 0.9.0, 2024-01-31

Code name: Dependency wrangling


## Summary

This release changes how we import the torch package, adds the sacremoses tokenizer 
and includes security updates. It also adds functions to load locally saved models as 
a preparation for changing the model downloading and saving process.

### Features

- #145: Added load function for loading local models

### Refactorings

 - #182: Removed torch package index from pyproject.toml
 - #139: Installs Sacremoses tokenizer

### Security 

  - Update gitpython (3.1.40 -> 3.1.41)
  - Updated jinja2 (3.1.2 -> 3.1.3)
