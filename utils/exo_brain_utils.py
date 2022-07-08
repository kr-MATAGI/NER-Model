import re
from dataclasses import dataclass
from typing import List


### Regex
RE_NE = r"<[^:]+[^>]+>"

### NE Tag
NE_TAG = {
    "O": 0,
    "DT": 1, "LC": 1, "OG": 2, "PS": 3, "TI": 4
}

### MAIN ###
if "__main__" == __name__:
    print(f"[exo_brain_utils] ---MAIN")

    data_path = "../data/exo_brain/EXOBRAIN_NE_CORPUS_10000.txt"

    text_lines: List[str] = []
    with open(data_path, mode="r", encoding="utf-8") as text_file:
        text_lines = text_file.readlines()
        text_lines = [x.replace("\n", "") for x in text_lines]
        print(f"[exo_brain_utils][main] READ text file path: {data_path}")
        print(f"[exo_brain_utils][main] readlines() size: {len(text_lines)}")

    for sent_idx, sent in enumerate(text_lines):
        print(sent)

        re_find_all = re.findall(RE_NE, sent)
        for ne_pair in re_find_all:
            split_res = ne_pair.split(":")
            ne = split_res[0]
            ne = ne[1:]
            label = split_res[-1]
            label = label[:-1]
            print(ne, label)

        break