# Using Prediction UDFs

We provide 7 prediction UDFs in the Transformers Extension package. Each performs an NLP task through the [transformers API](https://huggingface.co/docs/transformers/task_summary).  These tasks use the model downloaded to the BucketFS and run inference using the user-supplied inputs.

### Table of Contents

  * [AI Custom Classify Extended](#ai-custom-classify-extended)
  * [AI Entailment Extended](#ai-entailment-extended)
  * [Question Answering UDF](#question-answering-udf)
  * [Masked Language Modelling UDF](#masked-language-modelling-udf)
  * [Text Generation UDF](#text-generation-udf)
  * [Token Classification UDF](#token-classification-udf)
  * [Text Translation UDF](#text-translation-udf)
  * [Zero-Shot Text Classification UDF](#zero-shot-text-classification-udf)

    
### AI Custom Classify Extended

This UDF classifies the given text according to a given number of classes of the specified model.

Example usage:

```sql
SELECT AI_CUSTOM_CLASSIFY_EXTENDED(
    device_id,
    bucketfs_conn,
    sub_dir,
    model_name,
    text_data,
    return_ranks
)
```

[Common Parameters](./user_guide.md#common-udf-parameters)
* `device_id`
* `bucketfs_conn`
* `sub_dir`
* `model_name`

Specific parameters
* `text_data`: The input text to be classified
* `return_ranks`: String, either "ALL" which will result in all results being returned, or "HIGHEST", which will only return the result with rank=1 for this input.

Additional output columns
* _LABEL_: the predicted label for the input text
* _SCORE_: the confidence with which this label was assigned
* _RANK_: the rank of the label. In this context, all predictions/labels for one input are ranked by their score. rank=1 means best result/highest score.

Example:

| BUCKETFS_CONN | SUB_DIR | MODEL_NAME | TEXT_DATA | RETURN_RANKS | LABEL   | SCORE | RANK | ERROR_MESSAGE |
|---------------|---------|------------|-----------|--------------|---------|-------|------|---------------|
| conn_name     | dir/    | model_name | text 1    | ALL          | label_1 | 0.75  | 1    | None          |
| conn_name     | dir/    | model_name | text 1    | ALL          | label_2 | 0.23  | 2    | None          |
| ...           | ...     | ...        | ...       | ...          | ...     | ...   | ...  | ...           |

### AI Entailment Extended

This UDF takes two input texts and compares them. Among other things, it can be used to determine if two texts are paraphrases of each other.

Example usage:

```sql
SELECT AI_ENTAILMENT_EXTENDED(
    device_id,
    bucketfs_conn,
    sub_dir,
    model_name,
    first_text,
    second_text,
    return_ranks
)
```

[Common Parameters](./user_guide.md#common-udf-parameters)
* `device_id`
* `bucketfs_conn`
* `sub_dir`
* `model_name`

Specific parameters
* `first_text`: The first input text to be classified
* `second_text`: The second input text, a context for the first text
* `return_ranks`: String, either "ALL" which will result in all results being returned, or "HIGHEST", which will only return the result with rank=1 for this input.


Additional output columns
* _LABEL_: the predicted label for the input text
* _SCORE_: the confidence with which this label was assigned
* _RANK_: the rank of the label. In this context, all predictions/labels for one input are ranked by their score. rank=1 means best result/highest score.

| BUCKETFS_CONN | SUB_DIR | MODEL_NAME | FIRST_TEXT | SECOND_TEXT | RETURN_RANKS | LABEL   | SCORE | RANK | ERROR_MESSAGE |
|---------------|---------|------------|------------|-------------|--------------|---------|-------|------|---------------|
| conn_name     | dir/    | model_name | text 1     | text 2      | ALL          | label_1 | 0.75  | 1    | None          |
| conn_name     | dir/    | model_name | text 1     | text 2      | ALL          | label_2 | 0.23  | 2    | None          |
| ...           | ...     | ...        | ...        | ...         | ...          | ...     | ...   | ...  | ...           |


### Question Answering UDF

This UDF extracts answer(s) from a given question text. With the `top_k`
input parameter, up to `k` answers with the best inference scores can be returned.
An example usage is given below:

```sql
SELECT TE_QUESTION_ANSWERING_UDF(
    device_id,
    bucketfs_conn,
    sub_dir,
    model_name,
    question,
    context_text,
    top_k
)
```

[Common Parameters](./user_guide.md#common-udf-parameters)
* `device_id`
* `bucketfs_conn`
* `sub_dir`
* `model_name`

Specific parameters
* `question`: The question text
* `context_text`: The context text, associated with the question
* `top_k`: The number of answers to return.
   * Note that, `k` number of answers are not guaranteed.
   * If there are not enough options in the context, it might return less than `top_k` answers, see the [top_k parameter of QuestionAnswering](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.QuestionAnsweringPipeline.__call__).

Additional output columns
* _ANSWER_: the predicted answer for the input question
* _SCORE_: the confidence with which this answer was generated
* _RANK_: the rank of the answer. In this context, all predictions/labels for one input are ranked by their score. rank=1 means best result/highest score.

If `top_k` > 1, each input row is repeated for each answer.

Example:

| BUCKETFS_CONN | SUB_DIR | MODEL_NAME | QUESTION   | CONTEXT   | TOP_K | ANSWER   | SCORE | RANK | ERROR_MESSAGE |
|---------------|---------|------------|------------|-----------|-------|----------|-------|------|---------------|
| conn_name     | dir/    | model_name | question_1 | context_1 | 2     | answer_1 | 0.75  | 1    | None          |
| conn_name     | dir/    | model_name | question_2 | context_1 | 2     | answer_2 | 0.70  | 2    | None          |
| ...           | ...     | ...        | ...        | ...       | ...   | ...      | ...   | ..   | ...           |

### Masked Language Modelling UDF

This UDF needs to be given an inout text containing the ```<mask>``` token. It can then 
replace these masks with appropriate tokens. 
I.E the input text could be "<mask> is the best database Software for Machine 
Learning Enthusiasts.", resulting in an output like "Exasol is the best database 
Software for Machine Learning Enthusiasts."

Example usage:

```sql
SELECT AI_FILL_MASK_EXTENDED(
    device_id,
    bucketfs_conn,
    sub_dir,
    model_name,
    text_data,
    top_k
)
```

[Common Parameters](./user_guide.md#common-udf-parameters)
* `device_id`
* `bucketfs_conn`
* `sub_dir`
* `model_name`

Specific parameters
* `text_data`: The text data containing masking tokens
* `top_k`: The number of predictions to return.

Additional output columns
* _FILLED_TEXT_: the filled text (whole input text with mask token replaced)
* _SCORE_: the confidence with witch the mask was filled
* _RANK_: the rank of the answer. In this context, all predictions/labels for one input are ranked by their score. rank=1 means best result/highest score.

If `top_k` > 1, each input row is repeated for each prediction.

Example:

| BUCKETFS_CONN | SUB_DIR | MODEL_NAME | TEXT_DATA     | TOP_K | FILLED_TEXT   | SCORE | RANK | ERROR_MESSAGE |
| ------------- |---------|------------|---------------| ----- |---------------| ----- |------|---------------|
| conn_name     | dir/    | model_name | text `<mask>` | 2     | text filled_1 | 0.75  |   1  | None          |
| conn_name     | dir/    | model_name | text `<mask>` | 2     | text filled_2 | 0.70  |   2  | None          |
| ...           | ...     | ...        | ...           | ...   | ...           | ...   |  ... | ...           |

### Text Generation UDF

This UDF aims to consistently predict the continuation of the given text.  The length of the text to be generated is limited by the `max_new_tokens` parameter.

Example usage:

```sql
SELECT TE_TEXT_GENERATION_UDF(
    device_id,
    bucketfs_conn,
    sub_dir,
    model_name,
    text_data,
    max_new_tokens,
    return_full_text
)
```

[Common Parameters](./user_guide.md#common-udf-parameters)
* `device_id`
* `bucketfs_conn`
* `sub_dir`
* `model_name`

Specific parameters
* `text_data`: The context text.
* `max_new_tokens`: The maximum total number of tokens in the generated text.
* `return_full_text`:  If set to `FALSE`, only added text is returned, otherwise the full text is returned.

Additional output columns
* _GENERATED_TEXT_: the generated text

Example:

| BUCKETFS_CONN | SUB_DIR | MODEL_NAME | TEXT_DATA      | MAX_NEW_TOKENS | RETURN_FULL_TEXT | GENERATED_TEXT                          | ERROR_MESSAGE |
| ------------- |---------|------------|----------------|----------------|------------------|-----------------------------------------|---------------|
| conn_name     | dir/    | model_name | beginning text | 30             | True             | beginning text often includes a summary | None          |
| conn_name     | dir/    | model_name | continue       | 30             | False            | this sentence                           | None          |
| ...           | ...     | ...        | ...            | ...            | ...              | ...                                     | ...           |


### Token Classification UDF

The main goal of this UDF is to find tokens in a given text, and assign a label to found tokens.

There are two popular subtasks of token classification:

*  Named Entity Recognition (NER) which identifies specific entities in a text, such as dates, people, and places.
*  Part of Speech (PoS) which identifies which words in a text are verbs, nouns, and punctuation.

```sql
SELECT TE_TOKEN_CLASSIFICATION_UDF(
    device_id,
    bucketfs_conn,
    sub_dir,
    model_name,
    text_data,
    aggregation_strategy
)
```

[Common Parameters](./user_guide.md#common-udf-parameters)
* `device_id`
* `bucketfs_conn`
* `sub_dir`
* `model_name`

Specific parameters
* `text_data`: The text to analyze.
* `aggregation_strategy`: The strategy about whether to fuse tokens based on the model prediction. `NULL` means "simple" strategy, see [huggingface.co](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.TokenClassificationPipeline.aggregation_strategy) for more information.

Additional output columns
* _START_POS_: the index of the starting character of the found token
* _END_POS_: index of the ending character of the found token
* _WORD_: the found token
* _ENTITY_: the token-type the found token was sorted into.
* _SCORE_: the confidence with which the label was predicted

In case the model returns an empty result for an input row, the row is dropped entirely and not part of the result set.

Example:

| BUCKETFS_CONN | SUB_DIR | MODEL_NAME | TEXT_DATA | AGGREGATION_STRATEGY | START_POS | END_POS | WORD | ENTITY | SCORE | ERROR_MESSAGE |
| ------------- |---------|------------|-----------|----------------------|-----------|---------|------|--------|-------| ------------- |
| conn_name     | dir/    | model_name | text      | simple               | 0         | 4       | text | noun   | 0.75  | None          |
| ...           | ...     | ...        | ...       | ...                  | ...       | ...     | ...  | ..     | ...   | ...           |

### Text Translation UDF

This UDF translates a given text from one language to another.

```sql
SELECT TE_TRANSLATION_UDF(
    device_id,
    bucketfs_conn,
    sub_dir,
    model_name,
    text_data,
    source_language,
    target_language,
    max_new_tokens
)
```

[Common Parameters](./user_guide.md#common-udf-parameters)
* `device_id`
* `bucketfs_conn`
* `sub_dir`
* `model_name`

Specific parameters
* `text_data`: The text to translate.
* `source_language`: The language of the input. Required for multilingual models only. (see [Transformers Translation API](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.TranslationPipeline.__call__)).
* `target_language`:  The language of the desired output. Required for multilingual models only. (see [Transformers Translation API](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.TranslationPipeline.__call__)).
* `max_new_tokens`: The maximum total number of tokens in the translated text.

Additional output columns
* _TRANSLATION_TEXT_: the translated text

Example:

| BUCKETFS_CONN | SUB_DIR | MODEL_NAME | TEXT_DATA | SOURCE_LANGUAGE | TARGET_LANGUAGE | MAX_NEW_TOKENS | TRANSLATION_TEXT | ERROR_MESSAGE |
|---------------|---------|------------|-----------|-----------------|-----------------|------------|------------------|---------------|
| conn_name     | dir/    | model_name | context   | English         | German          | 100        | kontext          | None          |
| ...           | ...     | ...        | ...       | ...             | ...             | ...        | ...              | ...           |

### Zero-Shot Text Classification UDF

This UDF provides the task of predicting a class that was not seen by the model during training.

The UDF takes candidate labels as a comma-separated string and generates probability scores for each predicted label.


```sql
SELECT TE_ZERO_SHOT_TEXT_CLASSIFICATION_UDF(
    device_id,
    bucketfs_conn,
    sub_dir,
    model_name,
    text_data,
    candidate_labels,
    return_ranks
)
```

[Common Parameters](./user_guide.md#common-udf-parameters)
* `device_id`
* `bucketfs_conn`
* `sub_dir`
* `model_name`

Specific parameters
* `text_data`: The text to be classified.
* `candidate_labels`: A list of comma-separated labels, e.g. `label1,label2,label3`. Only these labels will be used in the prediction.
* `return_ranks`: String, either "ALL" which will result in all results being returned, or "HIGHEST", which will only return the result with rank=1 for this input.

Additional output columns
* _LABEL_: the predicted label for the input text
* _SCORE_: the confidence with witch the label was assigned
* _RANK_: the rank of the label. In this context, all predictions/labels for one input are ranked by their score. rank=1 means best result/highest score.

Example:

| BUCKETFS_CONN | SUB_DIR | MODEL_NAME | TEXT_DATA | CANDIDATE LABELS | RETURN_RANKS | LABEL  | SCORE | RANK | ERROR_MESSAGE |
| ------------- |---------|------------|-----------|------------------|--------------|--------|-------|------|---------------|
| conn_name     | dir/    | model_name | text      | label1,label2..  | ALL          | label1 | 0.75  | 1    | None          |
| conn_name     | dir/    | model_name | text      | label1,label2..  | ALL          | label2 | 0.70  | 2    | None          |
| ...           | ...     | ...        | ...       | ...              | ...          | ...    | ...   | ..   | ...           |
