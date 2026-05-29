text_data = "My test text"

token = "token"
token_start = 2
token_end = 4
entity_type = "ENTITY_TYPE"
score = 0.1
error_msg = None


def make_output_row(
    text_data=text_data,
    start=token_start,
    end=token_end,
    word=token,
    entity=entity_type,
    score=score,
    error_msg=error_msg,
):
    return [(text_data, start, end, word, entity, score, error_msg)]


def make_model_output_for_one_input_row(
    number_entities: int,
    entity_group=entity_type,
    score=score,
    word=token,
    start=token_start,
    end=token_end,
):
    model_output_single_entities = {
        "entity_group": entity_group,
        "score": score,
        "word": word,
        "start": start,
        "end": end,
    }
    return [[model_output_single_entities] * number_entities]
