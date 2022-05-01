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

def make_npy(mode: str, tokenizer_name: str, sent_list: List[Sentence]):
    '''
    Args:
        mode: "etri" / ...
        tokenizer_name: hugging face's tokenizer name
        sent_list: list of Sentence class
    '''

    print(f"[npy_maker][make_npy] Mode: {mode}, Sentence List Size: {len(sent_list)}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    for sent_item in sent_list:
        ret_dict = {
            "input_ids": [],
            "labels": [],
            "attention_mask": [],
            "token_type_ids": []
        }

        # ref morp
        token_id_pair_list = []
        for morp_item in sent_item.morp_list:
            token = [tok.replace("##", "") for tok in tokenizer.tokenize(morp_item.lemma)]
            token_id_pair_list.append((token, morp_item.id, ["O" for _ in range(len(token))]))

        # ref ne
        for ne_item in sent_item.ne_list:
            id_list = [x for x in range(ne_item.begin, ne_item.end+1)]

            if 1 == len(id_list):
                #print(id_list, token_id_pair_list[id_list[0]])
                for tag_idx in range(len((token_id_pair_list[id_list[0]][-1]))):
                    if 0 == tag_idx:
                        token_id_pair_list[id_list[0]][-1][tag_idx] = "B-" + ne_item.type
                    else:
                        token_id_pair_list[id_list[0]][-1][tag_idx] = "I-" + ne_item.type
                #print(id_list, token_id_pair_list[id_list[0]])
            else:
                for id_idx, id_item in enumerate(id_list):
                    #print(id_list, id_idx, token_id_pair_list[id_item])
                    if 0 == id_idx:
                        for tag_idx in range(len(token_id_pair_list[id_item][-1])):
                            if 0 == tag_idx:
                                token_id_pair_list[id_item][-1][tag_idx] = "B-" + ne_item.type
                            else:
                                token_id_pair_list[id_item][-1][tag_idx] = "I-" + ne_item.type
                    else:
                        for tag_idx in range(len(token_id_pair_list[id_item][-1])):
                            token_id_pair_list[id_item][-1][tag_idx] = "I-" + ne_item.type
                    #print(id_list, id_idx, token_id_pair_list[id_item])

        # make dict
        # 개체명이 있는 단어만 남겨둠 - morp만으로 text를 생성할 경우, raw text와 다름
        filter_token_ne_pair = list(filter(lambda x: True if "O" not in x[-1] else False, token_id_pair_list))
        tokens = tokenizer.tokenize(sent_item.text)

        pair_idx = 0
        for tok in tokens:
            conv_tok = tok.replace("##", "")
            if conv_tok in filter_token_ne_pair[pair_idx][0]:
                target_idx = filter_token_ne_pair[pair_idx][0].index(conv_tok)
                print(target_idx, conv_tok, filter_token_ne_pair[pair_idx], pair_idx)
                if target_idx == (len(filter_token_ne_pair[pair_idx][0]) - 1):
                    pair_idx += 1
        print(filter_token_ne_pair)
        break


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