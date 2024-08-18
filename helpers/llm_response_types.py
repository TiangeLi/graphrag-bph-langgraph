from typing import TypedDict, List

class BooleanResponse(TypedDict):
    b: bool

class StringResponse(TypedDict):
    s: str

class ListOfStringsResponse(TypedDict):
    l: List[str]