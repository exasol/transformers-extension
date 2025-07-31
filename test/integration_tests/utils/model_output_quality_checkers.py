
def assert_lenient_check_of_output_quality_with_score(
    result: list, n_rows_result: int, acceptable_results: list,
        acceptance_factor: float, label_index: int = 5,
):
    """
    Checks whether enough of the results are of "good quality".
    We do this by seeing if the result label is one of our predefined "acceptable_results", and how high the score is.
    We want high confidence on good results, and low confidence on bad results. however, cutoffs for
    high and low confidence, as well as defined "acceptable_results" where not set in an elaborate scientific way.
    This check is only here to assure us the models output is not totally of kilter
    (and crucially does not get worse with our changes over time),
    and therefore we can assume model loading and execution is working correctly.
    We plan to make this check deterministic in the future.

    An accepted result is defined as follows:
                         | label acceptable  | label unacceptable
    --------------------------------------------------------------
    high confidence       | acceptable        |  bad result
    (result_score > 0.8)  |                   |
    --------------------------------------------------------------
    other confidence      | result not        |  result not
    (result_score between | good enough to    |  good enough to
    high and low)         | be accepted       |  be accepted
    --------------------------------------------------------------
    low confidence        | bad result        |  acceptable
    (result_score < 0.2)  |                   |

    We only sum up acceptable results below, because we already know we
    have the correct number of results from the other checks.
    """

    def contains(string, list):
        return any(map(lambda x: x in string, list))

    number_accepted_results = 0
    for result_i in result:
        result_label = result_i[label_index]
        result_score = result_i[label_index + 1]
        if (
            contains(result_label, acceptable_results) and result_score > 0.8
        ):  # check if confidence on good results is reasonably high
            number_accepted_results += 1
        elif result_score < 0.2 and not contains(result_label, acceptable_results):
            number_accepted_results += 1
    assert (
        number_accepted_results > n_rows_result / acceptance_factor
    ), f"Not enough acceptable labels ({acceptable_results}) in results {result}"

def assert_lenient_check_of_output_quality(
        result: list, n_rows_result: int, acceptable_results: list,
        acceptance_factor: float, label_index: int = 5,
):
    """
    Lenient test for quality of results.
    We do this by seeing if the result is one of our predefined "acceptable_results".
    This check is only here to assure us the models output is not totally of kilter
    (and crucially does not get worse with our changes over time),
    and therefore we can assume model loading and execution is working correctly.
    We plan to make this check deterministic in the future.
    """
    results = [result[i][label_index] for i in range(len(result))]
    number_accepted_results = 0

    def contains(string, list):
        return any(map(lambda x: x in string, list))

    for i in range(len(results)):
        if contains(results[i], acceptable_results):
            number_accepted_results += 1
    assert number_accepted_results > n_rows_result / acceptance_factor

def assert_lenient_check_of_output_quality_for_result_set(
        result: list, acceptable_result_sets: list[list],
        acceptance_factor: float, label_index: int = 5,
):
    """
    Lenient test for quality of results.
    We do this by seeing if the result is one of our predefined "acceptable_results".
    This check is only here to assure us the models output is not totally of kilter
    (and crucially does not get worse with our changes over time),
    and therefore we can assume model loading and execution is working correctly.
    We to make this check deterministic in the future.
    """
    results = [[result[i][label_index], result[i][label_index+1]] for i in range(len(result))]
    number_accepted_results = 0

    for i in range(len(results)):
        if results[i] in acceptable_result_sets:
            number_accepted_results += 1
    assert (
        number_accepted_results > len(results) / acceptance_factor
    ), f"Not enough acceptable results {acceptable_result_sets} in results {results}"
