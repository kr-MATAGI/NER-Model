import pickle
import copy
from typing import List

from data_def import Sentence, Morp, NE, Word
from transformers import AutoTokenizer
from tokenizers

### MAIN ##
if "__main__" == __name__:
    src_file_path = "../data/pkl/ne_mp_old_nikl.pkl"
    tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-small-v3-discriminator")
    results = { # tag: (word_item, morp_item_list) pair
        "JX": [],
        "EP": [],
        "EF": [],
        "EC": [],
        "ETM": [],
    }
    src_list: List[Sentence] = []
    with open(src_file_path, mode="rb") as load_pkl:
        src_list = pickle.load(load_pkl)
        print(f"LOAD SIZE: {len(src_list)}")

    target_pos_list = results.keys()
    end_point_pos = ["SF", "SP", "SS", "SE", "SO", "SW"]

    for src_idx, src_item in enumerate(src_list):
        if 0 == (src_idx % 1000):
            print(f"{src_idx} is Processing... {src_item.text}")

        new_morp_list: List[Morp] = []
        ignore_morp_id = []

        for morp_item_idx, morp_item in enumerate(src_item.morp_list):
            if morp_item.id in ignore_morp_id:
                continue

