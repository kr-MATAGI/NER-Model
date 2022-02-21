from dataclasses import dataclass, field
from typing import List

TTA_NE_tags = {
    "O": 0,
    "B-PS": 1, "I-PS": 2, # person
    "B-LC": 3, "I-LC": 4, # location
    "B-OG": 5, "I-OG": 6, # organization
    "B-AF": 7, "I-AF": 8, # artifacts
    "B-DT": 9, "I-DT": 10, # date
    "B-TI": 11, "I-TI": 12, # time
    "B-CV": 13, "I-CV": 14, # civilization
    "B-AM": 15, "I-AM": 16, # animal
    "B-PT": 17, "I-PT": 18, # plant
    "B-QT": 19, "I-QT": 20, # quantity
    "B-FD": 21, "I-FD": 22, # study_field
    "B-TR": 23, "I-TR": 24, # theory
    "B-EV": 25, "I-EV": 26, # event
    "B-MT": 27, "I-MT": 28, # material
    "B-TM": 29, "I-TM": 30 # term
}

@dataclass
class EXO_NE:
    id: int = -1
    text: str = ""
    type: str = ""
    begin: int = -1
    end: int = -1
    weight: float = -1.0
    common_noun: int = -1

@dataclass
class EXO_NE_Json:
    sent_id: int = -1
    text: str = ""
    ne_list: List[EXO_NE] = field(default_factory=list)