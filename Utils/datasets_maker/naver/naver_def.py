from dataclasses import dataclass, field
from typing import List

@dataclass
class NAVER_NE:
    sentence: str = ""
    word_list: List[str] = field(default_factory=list)
    tag_list: List[str] = field(default_factory=list)

NAVER_NE_MAP = {
    "O": 0,
    "B-PER": 1, "I-PER": 2, # 인물
    "B-FLD": 3, "I-FLD": 4, # 학문 분야
    "B-AFW": 5, "I-AFW": 6, # 인공물
    "B-ORG": 7, "I-ORG": 8, # 기관 및 단체
    "B-LOC": 9, "I-LOC": 10, # 지역명
    "B-CVL": 11, "I-CVL": 12, # 문명 및 문화
    "B-DAT": 13, "I-DAT": 14, # 날짜
    "B-TIM": 15, "I-TIM": 16, # 시간
    "B-NUM": 17, "I-NUM": 18, # 숫자
    "B-EVT": 19, "I-EVT": 20, # 사건사고 및 행사
    "B-ANM": 21, "I-ANM": 22, # 동물
    "B-PLT": 23, "I-PLT": 24, # 식물
    "B-MAT": 25, "I-MAT": 26, # 금속/암석/화학물질
    "B-TRM": 27, "I-TRM": 28, # 의학용어/IT관련 용어
    "X": 29, # special token
}