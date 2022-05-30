import copy
import pickle
from builtins import enumerate, str

import numpy as np
import torch
from typing import List
from eunjeon import Mecab # window
from konlpy.tag import Mecab # mac

from transformers import AutoTokenizer
from data_def import *
from pos_tag_def import MECAB_POS_TAG

ETRI_TAG = {
    "O": 0,
    "B-PS": 1, "I-PS": 2,
    "B-LC": 3, "I-LC": 4,
    "B-OG": 5, "I-OG": 6,
    "B-AF": 7, "I-AF": 8,
    "B-DT": 9, "I-DT": 10,
    "B-TI": 11, "I-TI": 12,
    "B-CV": 13, "I-CV": 14,
    "B-AM": 15, "I-AM": 16,
    "B-PT": 17, "I-PT": 18,
    "B-QT": 19, "I-QT": 20,
    "B-FD": 21, "I-FD": 22,
    "B-TR": 23, "I-TR": 24,
    "B-EV": 25, "I-EV": 26,
    "B-MT": 27, "I-MT": 28,
    "B-TM": 29, "I-TM": 30
}

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

def make_npy(mode: str, tokenizer_name: str, sent_list: List[Sentence], max_len: int=512,
             use_pos: bool=False):
    '''
    Args:
        mode: "etri" / ...
        tokenizer_name: hugging face's tokenizer name
        sent_list: list of Sentence class
    '''

    print(f"[npy_maker][make_npy] Mode: {mode}, Sentence List Size: {len(sent_list)}")

    npy_dict = {
        "input_ids": [],
        "labels": [],
        "attention_mask": [],
        "token_type_ids": [],
        "seq_len": [],
        "pos_tag": [],
    }
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if use_pos:
        mecab = Mecab()
        mecab_pos2ids = {v: k for k, v in MECAB_POS_TAG.items()}
    for proc_idx, sent in enumerate(sent_list):
        if 0 == (proc_idx % 1000):
            print(f"{proc_idx} Processing... {sent.text}")
        text_tokens = tokenizer.tokenize(sent.text)
        labels = ["O"] * len(text_tokens)

        if use_pos:
            # additional pos tag info
            # target_ne_type_set = ["FD", "AF", "TM", "AM"] # f1: 0.84, 0.73, 0.83, 0.87
            pos_tag_vec = [[0 for _ in range(3)] for _ in range(max_len)] # [seq_len, 3]
            sent_pos_res = mecab.pos(sent.text)
            start_idx = 0

            prev_rhs = ""
            for lhs, rhs in sent_pos_res: # [형태소, pos]
                prev_rhs += lhs
                concat_word = ""
                for s_idx in range(start_idx, len(text_tokens)):
                    concat_word += text_tokens[s_idx].replace("##", "")
                    if concat_word == prev_rhs:
                        sp_rhs = rhs.split("+")
                        pos_tag_elem = [0 for _ in range(3)]
                        for sp_idx, sp_item in enumerate(sp_rhs):
                            if 3 <= sp_idx:
                                break
                            if sp_item not in mecab_pos2ids.keys():
                                continue
                            conv_ids = mecab_pos2ids[sp_item]
                            pos_tag_elem[sp_idx] = conv_ids
                        for vec_idx in range(start_idx, s_idx+1):
                            if max_len <= vec_idx:
                                break
                            pos_tag_vec[vec_idx] = copy.deepcopy(pos_tag_elem)
                        concat_word = ""
                        prev_rhs = ""
                        start_idx = s_idx + 1
                        break
            npy_dict["pos_tag"].append(pos_tag_vec)

        start_idx = 0
        for ne_item in sent.ne_list:
            ne_tokens = tokenizer.tokenize(ne_item.text)

            for s_idx in range(start_idx, len(text_tokens)):
                concat_token = text_tokens[s_idx:s_idx+len(ne_tokens)]

                if "".join(concat_token) == "".join(ne_tokens):
                    for l_idx in range(s_idx, s_idx+len(ne_tokens)):
                        if l_idx == s_idx:
                            labels[l_idx] = "B-" + ne_item.type
                        else:
                            labels[l_idx] = "I-" + ne_item.type
                    start_idx = s_idx
                    break
        ## end, ne_item loop

        # TEST
        # test_ne_print = [(x.text, x.type) for x in sent.ne_list]
        # id2la = {v: k for k, v in ETRI_TAG.items()}
        # print(test_ne_print)
        # for t, l in zip(text_tokens, labels):
        #     print(t, "\t", l)
        # input(11111111)

        text_tokens.insert(0, "[CLS]")
        labels.insert(0, "O")

        valid_len = 0
        if max_len <= len(text_tokens):
            text_tokens = text_tokens[:max_len-1]
            labels = labels[:max_len-1]

            text_tokens.append("[SEP]")
            labels.append("O")

            valid_len = max_len
        else:
            text_tokens.append("[SEP]")
            labels.append("O")
            valid_len = len(text_tokens)
            text_tokens = text_tokens + ["[PAD]"] * (max_len - valid_len)
            labels = labels + ["O"] * (max_len - valid_len)

        attention_mask = ([1] * valid_len) + ([0] * (max_len - valid_len))
        token_type_ids = [0] * max_len
        input_ids = tokenizer.convert_tokens_to_ids(text_tokens)
        labels = [ETRI_TAG[x] for x in labels]

        assert len(input_ids) == max_len, f"{input_ids} + {len(input_ids)}"
        assert len(labels) == max_len, f"{labels} + {len(labels)}"
        assert len(attention_mask) == max_len, f"{attention_mask} + {len(attention_mask)}"
        assert len(token_type_ids) == max_len, f"{token_type_ids} + {len(token_type_ids)}"

        npy_dict["input_ids"].append(input_ids)
        npy_dict["labels"].append(labels)
        npy_dict["attention_mask"].append(attention_mask)
        npy_dict["token_type_ids"].append(token_type_ids)

        # for pack_padded_sequence
        npy_dict["seq_len"].append(valid_len)

    # convert list to numpy
    npy_dict["input_ids"] = np.array(npy_dict["input_ids"])
    npy_dict["labels"] = np.array(npy_dict["labels"])
    npy_dict["attention_mask"] = np.array(npy_dict["attention_mask"])
    npy_dict["token_type_ids"] = np.array(npy_dict["token_type_ids"])
    npy_dict["seq_len"] = np.array(npy_dict["seq_len"])
    npy_dict["pos_tag"] = np.array(npy_dict["pos_tag"])
    print(f"input_ids.shape: {npy_dict['input_ids'].shape}")
    print(f"labels.shape: {npy_dict['labels'].shape}")
    print(f"attention_mask.shape: {npy_dict['attention_mask'].shape}")
    print(f"token_type_ids.shape: {npy_dict['token_type_ids'].shape}")
    print(f"seq_len.shape: {npy_dict['seq_len'].shape}")
    print(f"pos_tag.shape: {npy_dict['pos_tag'].shape}")

    split_size = int(len(sent_list) * 0.1)
    train_size = split_size * 7
    valid_size = train_size + split_size

    train_np = [npy_dict["input_ids"][:train_size], npy_dict["labels"][:train_size],
                npy_dict["attention_mask"][:train_size], npy_dict["token_type_ids"][:train_size]]
    train_np = np.stack(train_np, axis=-1)
    train_seq_len_np = npy_dict["seq_len"][:train_size]
    train_pos_tag_np = npy_dict["pos_tag"][:train_size]
    print(f"train_np.shape: {train_np.shape}")
    print(f"train_seq_len_np.shape: {train_seq_len_np.shape}")
    print(f"train_pos_tag_np.shape: {train_pos_tag_np.shape}")

    dev_np = [npy_dict["input_ids"][train_size:valid_size], npy_dict["labels"][train_size:valid_size],
              npy_dict["attention_mask"][train_size:valid_size], npy_dict["token_type_ids"][train_size:valid_size]]
    dev_np = np.stack(dev_np, axis=-1)
    dev_seq_len_np = npy_dict["seq_len"][train_size:valid_size]
    dev_pos_tag_np = npy_dict["pos_tag"][train_size:valid_size]
    print(f"dev_np.shape: {dev_np.shape}")
    print(f"dev_seq_len_np.shape: {dev_seq_len_np.shape}")
    print(f"dev_pos_tag_np.shape: {dev_pos_tag_np.shape}")

    test_np = [npy_dict["input_ids"][valid_size:], npy_dict["labels"][valid_size:],
               npy_dict["attention_mask"][valid_size:], npy_dict["token_type_ids"][valid_size:]]
    test_np = np.stack(test_np, axis=-1)
    test_seq_len_np = npy_dict["seq_len"][valid_size:]
    test_pos_tag_np = npy_dict["pos_tag"][valid_size:]
    print(f"test_np.shape: {test_np.shape}")
    print(f"test_seq_len_np.shape: {test_seq_len_np.shape}")
    print(f"test_pos_tag_np.shape: {test_pos_tag_np.shape}")

    # save
    np.save("../data/npy/"+mode+"/128/train", train_np)
    np.save("../data/npy/"+mode+"/128/dev", dev_np)
    np.save("../data/npy/"+mode+"/128/test", test_np)

    # save seq_len
    np.save("../data/npy/" + mode + "/128/train_seq_len", train_seq_len_np)
    np.save("../data/npy/" + mode + "/128/dev_seq_len", dev_seq_len_np)
    np.save("../data/npy/" + mode + "/128/test_seq_len", test_seq_len_np)

    # save pos_tag
    np.save("../data/npy/" + mode + "/128/train_pos_tag", train_pos_tag_np)
    np.save("../data/npy/" + mode + "/128/dev_pos_tag", dev_pos_tag_np)
    np.save("../data/npy/" + mode + "/128/test_pos_tag", test_pos_tag_np)

### MAIN ###
if "__main__" == __name__:
    print(f"[npy_maker][__main__] NPY MAKER !")

    all_sent_list = []
    with open("../data/pkl/old_nikl.pkl", mode="rb") as pkl_file:
        all_sent_list = pickle.load(pkl_file)
        print(f"[npy_maker][__main__] all_sent_list size: {len(all_sent_list)}")
    all_sent_list = conv_TTA_ne_category(all_sent_list)

    # make npy
    make_npy(mode="old_nikl", tokenizer_name="klue/bert-base", sent_list=all_sent_list,
             max_len=128, use_pos=True)
