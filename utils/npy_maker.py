import pickle
import numpy as np
import torch
from typing import List

from transformers import AutoTokenizer
from data_def import *

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

def make_npy(mode: str, tokenizer_name: str, sent_list: List[Sentence], max_len: int=512):
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
        "token_type_ids": []
    }
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    for sent in sent_list:
        text_tokens = tokenizer.tokenize(sent.text)
        labels = ["O"] * len(text_tokens)

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
        test_ne_print = [(x.text, x.type) for x in sent.ne_list]
        id2la = {v: k for k, v in ETRI_TAG.items()}
        print(test_ne_print)
        for t, l in zip(text_tokens, labels):
            print(t, "\t", l)
        input()

        text_tokens.insert(0, "[CLS]")
        labels.insert(0, "O")

        valid_len = 0
        if max_len <= len(text_tokens):
            text_tokens = text_tokens[:max_len-1]
            labels = labels[:max_len-1]
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

        npy_dict["input_ids"].append(input_ids)
        npy_dict["labels"].append(labels)
        npy_dict["attention_mask"].append(attention_mask)
        npy_dict["token_type_ids"].append(token_type_ids)

    # convert list to numpy
    npy_dict["input_ids"] = np.array(npy_dict["input_ids"])
    npy_dict["labels"] = np.array(npy_dict["labels"])
    npy_dict["attention_mask"] = np.array(npy_dict["attention_mask"])
    npy_dict["token_type_ids"] = np.array(npy_dict["token_type_ids"])
    print(f"input_ids.shape: {npy_dict['input_ids'].shape}")
    print(f"labels.shape: {npy_dict['labels'].shape}")
    print(f"attention_mask.shape: {npy_dict['attention_mask'].shape}")
    print(f"token_type_ids.shape: {npy_dict['token_type_ids'].shape}")

    split_size = int(len(sent_list) * 0.1)
    train_size = split_size * 7
    valid_size = train_size + split_size

    train_np = [npy_dict["input_ids"][:train_size], npy_dict["labels"][:train_size],
                npy_dict["attention_mask"][:train_size], npy_dict["token_type_ids"][:train_size]]
    train_np = np.stack(train_np, axis=-1)
    print(f"train_np.shape: {train_np.shape}")

    valid_np = [npy_dict["input_ids"][train_size:valid_size], npy_dict["labels"][train_size:valid_size],
                npy_dict["attention_mask"][train_size:valid_size], npy_dict["token_type_ids"][train_size:valid_size]]
    valid_np = np.stack(valid_np, axis=-1)
    print(f"valid_np.shape: {valid_np.shape}")

    test_np = [npy_dict["input_ids"][valid_size:], npy_dict["labels"][valid_size:],
               npy_dict["attention_mask"][valid_size:], npy_dict["token_type_ids"][valid_size:]]
    test_np = np.stack(test_np, axis=-1)
    print(f"test_np.shape: {test_np.shape}")

    # save
    np.save("../data/npy/etri/train", train_np)
    np.save("../data/npy/etri/dev", valid_np)
    np.save("../data/npy/etri/test", test_np)

### MAIN ###
if "__main__" == __name__:
    print(f"[npy_maker][__main__] NPY MAKER !")

    all_sent_list = []
    with open("../data/pkl/etri.pkl", mode="rb") as pkl_file:
        all_sent_list = pickle.load(pkl_file)
        print(f"[npy_maker][__main__] all_sent_list size: {len(all_sent_list)}")
    all_sent_list = conv_TTA_ne_category(all_sent_list)

    # make npy
    make_npy(mode="etri", tokenizer_name="klue/bert-base", sent_list=all_sent_list, max_len=30)