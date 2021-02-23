from typing import Any, List


def to_list(obj: Any) -> List:
    """
    Transform given object to list if obj is not already a list. Sets are also transformed to a list.

    :param obj: object to transform to list

    :return: list containing obj, or obj itself (if obj was already a list)
    """
    if isinstance(obj, (set, tuple)):
        obj = list(obj)
    elif not isinstance(obj, list):
        obj = [obj]
    return obj

