# Transformers Extension 1.0.0, 2024-04-05

Code name: Local model loading


## Summary

In this Release we integrated new model load functionality which means downloaded models will now be saved 
in the bucketFS, meaning you can now use pretrained models without connecting to the internet as 
long as you have saved them previously. There are also documentation updates.

### Breaking API changes

The change in the model loading functionality means the API of calling a model has changed. 
The 'token_conn' parameter was removed from the UDF calls. You can now call the UDFs 
as follows (Example case for the filling mask udf):

```sql
SELECT TE_FILLING_MASK_UDF(
    device_id,
    bucketfs_conn,
    sub_dir,
    model_name,
    text_data,
    top_k
)
```


### Features

- #146: Integrated new download and load functions using save_pretrained

### Documentation

- #133: Improved user and developer documentation with additional information

### Refactorings

- #147: Removed token_con from Prediction UDFs

