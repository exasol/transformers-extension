from typing import (
    Any,
    List,
    Tuple,
)


def python_value_to_sql(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, str):
        return f"'{value}'"
    if isinstance(value, int):
        return str(value)
    raise TypeError(f"Type {type(value)} of value {value} not supported.")


def python_row_to_sql(row: tuple) -> str:
    sql_values = [python_value_to_sql(value) for value in row]
    sql_row_str = ",".join(sql_values)
    return f"({sql_row_str})"


def python_rows_to_sql(rows: list[tuple]) -> str:
    sql_rows = [python_row_to_sql(row) for row in rows]
    sql_rows_str = ",".join(sql_rows)
    return sql_rows_str
