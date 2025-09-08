from collections.abc import Iterable


class AnyOrder:
    def __init__(self, expected: Iterable):
        self._expected = expected

    def __eq__(self, other: Iterable) -> bool:
        if not isinstance(other, Iterable):
            return False
        expected_list = list(self._expected)
        other_list = list(other)
        other_in_expected = all(item in expected_list for item in other_list)
        expected_in_other = all(item in other_list for item in expected_list)
        return other_in_expected and expected_in_other

    def __repr__(self) -> str:
        joined = ",\n ".join(str(item) for item in self._expected)
        return f"[{joined}]"
