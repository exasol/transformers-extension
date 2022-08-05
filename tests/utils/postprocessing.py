from typing import List
from exasol_udf_mock_python.group import Group


def get_rounded_result(result: List[Group], round_: int = 2) -> List[tuple]:
    rounded_result = result[0].rows
    for i in range(len(rounded_result)):
        rounded_result[i] = rounded_result[i][:-1] + \
                            (round(rounded_result[i][-1], round_),)
    return rounded_result

