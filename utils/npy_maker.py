import pickle
import numpy as np
import torch
from typing import List

from transformers import AutoTokenizer
from data_def import Sentence, Morp, NE

def conv_TTA_ne_category(sent_list: List[Sentence]):
    '''
    NE Type : 세분류 -> 대분류

    Args:
        sent_list: list of Sentence dataclass
    '''
    for sent_item in sent_list:
        for ne_item in sent_item.ne_list:
            conv_type = ""
            lhs_type = ne_item.type.split("_")[0]
            if "PS" in lhs_type:
                conv_type = "PS"
            elif "LC" in lhs_type:
                conv_type = "LC"
            elif "OG" in lhs_type:
                conv_type = "OG"
            elif "AF" in lhs_type:
                conv_type = "AF"
            elif "DT" in lhs_type:
                conv_type = "DT"
            elif "TI" in lhs_type:
                conv_type = "TI"
            elif "CV" in lhs_type:
                conv_type = "CV"
            elif "AM" in lhs_type:
                conv_type = "AM"
            elif "PT" in lhs_type:
                conv_type = "PT"
            elif "QT" in lhs_type:
                conv_type = "QT"
            elif "FD" in lhs_type:
                conv_type = "FD"
            elif "TR" in lhs_type:
                conv_type = "TR"
            elif "EV" in lhs_type:
                conv_type = "EV"
            elif "MT" in lhs_type:
                conv_type = "MT"
            elif "TM" in lhs_type:
                conv_type = "TM"

            if "" == conv_type:
                print(sent_item.text, "\n", lhs_type)
                return
            ne_item.type = conv_type
    return sent_list

def make_npy(mode: str, tokenizer_name: str, sent_list: List[Sentence], max_len: int=512):
    '''
    Args:
        mode: "etri" / ...
        tokenizer_name: hugging face's tokenizer name
        sent_list: list of Sentence class
    '''

    print(f"[npy_maker][make_npy] Mode: {mode}, Sentence List Size: {len(sent_list)}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    for sent in sent_list:
        print(sent.text)
        print(sent.ne_list)

        word_tokens = tokenizer.tokenize(sent.text)
        labels = ["O"] * len(word_tokens)

        start_idx = 0
        for ne_item in sent.ne_list:
            ne_tokens = tokenizer.tokenize(ne_item.text)

            is_find = False
            while True:
                concat_tokens = []
                for target_idx in range(start_idx, len(word_tokens)):
                    if target_idx >= len(word_tokens):
                        break
                    if word_tokens[target_idx] in ne_tokens:
                        concat_tokens.append(word_tokens[target_idx])
                    else:
                        start_idx += 1
                        break

                    if "".join(concat_tokens) == "".join(ne_tokens):
                        for label_idx in range(start_idx, target_idx+1):
                            if label_idx == start_idx:
                                labels[label_idx] = "B-" + ne_item.type
                            else:
                                labels[label_idx] = "I-" + ne_item.type
                        start_idx = target_idx
                        is_find = True
                        break
                    if is_find:
                        break
                if is_find:
                    break
        ## end, ne_item loop

        #
        word_tokens.insert(0, "[CLS]")
        labels.insert(0, "X")

        for t, l in zip(word_tokens, labels):
            print(t, l)
        input()

        valid_len = 0
        if max_len <= len(word_tokens):
            word_tokens = word_tokens[:max_len-1]
            labels = labels[:max_len-1]
            valid_len = 512
        else:
            word_tokens.append("[SEP]")
            labels.append("X")
            valid_len = len(word_tokens)
            word_tokens = word_tokens + ["[PAD]"] * (max_len - valid_len)
            labels = labels + ["X"] * (max_len - valid_len)

        attention_mask = ([1] * valid_len) + ([0] * (max_len - valid_len))
        token_type_ids = [0] * max_len


### MAIN ###
if "__main__" == __name__:
    print(f"[npy_maker][__main__] NPY MAKER !")

    all_sent_list = []
    with open("../data/pkl/etri.pkl", mode="rb") as pkl_file:
        all_sent_list = pickle.load(pkl_file)
        print(f"[npy_maker][__main__] all_sent_list size: {len(all_sent_list)}")
    all_sent_list = conv_TTA_ne_category(all_sent_list)

    # make npy
    make_npy(mode="etri", tokenizer_name="klue/bert-base", sent_list=all_sent_list)