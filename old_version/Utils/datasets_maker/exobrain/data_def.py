from dataclasses import dataclass, field
from typing import List

@dataclass
class NE:
    id: int = -1
    text: str = ""
    type: str = ""
    begin: int = -1
    end: int = -1
    weight: float = -1.0
    common_noun: int = -1

@dataclass
class EXO_Json:
    sent_id: str = ""
    text: str = ""
    ne_list: List[NE] = field(default_factory=list)