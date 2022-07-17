import copy
import re
import pickle
from dataclasses import dataclass, field
from typing import List

### Dataclass
@dataclass
class NE_Pair:
    text: str = ""
    label: str = "O"

@dataclass
class EXO_NE_Corpus:
    sent: str = ""
    ne_pair_list: List[NE_Pair] = field(default_factory=list)

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

    all_data_list: List[EXO_NE_Corpus] = []
    all_label_count: int = 0
    for sent_idx, sent in enumerate(text_lines):
        exo_ne_corpus = EXO_NE_Corpus()

        new_sent = copy.deepcopy(sent)
        re_find_all = re.findall(RE_NE, sent)
        all_label_count += len(re_find_all)
        for re_data in re_find_all:
            split_res = re_data.split(":")
            ne = split_res[0]
            ne = ne[1:]
            label = split_res[-1]
            label = label[:-1]

            ne_pair = NE_Pair(text=ne, label=label)
            new_sent = new_sent.replace(re_data, ne)

            exo_ne_corpus.sent = new_sent
            exo_ne_corpus.ne_pair_list.append(ne_pair)
        all_data_list.append(exo_ne_corpus)
    print(f"[exo_brain_utils][main] Complete Size: {len(all_data_list)}")
    print(f"[exo_brain_utils][main] all_label Size: {all_label_count}")

    # Save File
    save_path = "../data/exo_brain/EXOBRAIN_NE_CORPUS_10000.pkl"
    with open(save_path, mode="wb") as save_file:
        pickle.dump(all_data_list, save_file)
        print(f"[exo_brain_utils][main] Complete Save path - {save_path}")
