# Transformers Extension 1.0.0, 2024-04-12

Code name: Local model loading


## Summary

In this release, we integrated a new model loading functionality which means downloaded models will now be saved 
in the BucketFS. This means, the Prediction UDFs do not connect to the internet to look for model updates. 
There are also documentation updates, and we updated cryptography to >= 42.0.4.

### Breaking API changes

The change in the model loading functionality means the API for the Prediction UDFs has changed. 
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

- #205: Added vagrant setup
- #146: Integrated new download and load functions using save_pretrained

### Documentation

- #133: Improved user and developer documentation with additional information

### Refactorings

- #147: Removed token_con from Prediction UDFs

### Security 

 - Updated cryptography to >= 42.0.4

