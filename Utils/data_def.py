from dataclasses import dataclass, field
from enum import Enum
from typing import List

class TTA_NE_Tags(Enum):
    PS = "PS_" # Person
    LC = "LC_" # Location
    OG = "OG_" # Organization
    AF = "AF_" # Artifacts
    DT = "DT_" # Date
    TI = "TI_" # Time
    CV = "CV_" # Civilization
    AM = "AM_" # Animal
    PT = "PT_" # Plant
    QT = "QT_" # Quantity
    FD = "FD_" # Study_FIELD
    TR = "TR_" # Theory
    EV = "EV_" # Event
    MT = "MT_" # Material
    TM = "TM_" # Term

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
