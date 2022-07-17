import pickle
import numpy as np
import torch
from typing import List

from tokenization_kocharelectra import KoCharElectraTokenizer

import sys
sys.path.append("/Users/matagi/Desktop/Git/NER-Model/utils") # mac
from utils.data_def import *



#================================================
# Global
MODEL_NAME = "monologg/kocharelectra-base-discriminator"

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

#================================================
# Method
def make_char_npy(mode: str, tokenizer_name: str, src_item_list: List[Sentence],
                  max_len: int=128, use_pos=False):
    '''
        Args:
            mode: "etri" / ...
            tokenizer_name: hugging face's tokenizer name
            sent_list: list of Sentence class
    '''
    tokenizer = KoCharElectraTokenizer.from_pretrained(MODEL_NAME)

    npy_dict = {
        "input_ids": [],
        "labels": [],
        "attention_mask": [],
        "token_type_ids": [],
        "seq_len": []
    }

    for src_idx, src_item in enumerate(src_item_list):
        text_tokens = tokenizer.tokenize(src_item.text)
        labels = ["O"] * len(text_tokens)

        start_idx = 0
        for ne_idx, ne_item in enumerate(src_item.ne_list):
            ne_tokens = tokenizer.tokenize(ne_item.text)

            for s_idx in range(start_idx, len(text_tokens)):
                concat_token = text_tokens[s_idx:s_idx+len(ne_tokens)]

                if "".join(concat_token) == "".join(ne_tokens):
                    for label_idx in range(s_idx, s_idx+len(ne_tokens)):
                        if label_idx == s_idx:
                            labels[label_idx] = "B-" + ne_item.type
                        else:
                            labels[label_idx] = "I-" + ne_item.type
                    start_idx = s_idx
                    break
        #end, ne_item loop

        # # TEST
        test_ne_print = [(x.text, x.type) for x in src_item.ne_list]
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

            text_tokens.append("[SEP]")
            labels.append("O")

            valid_len = max_len
        else:
            text_tokens.append("[SEP]")
            labels.append("O")

            valid_len = len(text_tokens)
            text_tokens += ["[PAD]"] * (max_len - valid_len)
            labels += ["O"] * (max_len - valid_len)

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
    print(f"input_ids.shape: {npy_dict['input_ids'].shape}")
    print(f"labels.shape: {npy_dict['labels'].shape}")
    print(f"attention_mask.shape: {npy_dict['attention_mask'].shape}")
    print(f"token_type_ids.shape: {npy_dict['token_type_ids'].shape}")
    print(f"seq_len.shape: {npy_dict['seq_len'].shape}")

    split_size = int(len(src_item_list) * 0.1)
    train_size = split_size * 7
    valid_size = train_size + split_size

    train_np = [npy_dict["input_ids"][:train_size], npy_dict["labels"][:train_size],
                npy_dict["attention_mask"][:train_size], npy_dict["token_type_ids"][:train_size]]
    train_np = np.stack(train_np, axis=-1)
    train_seq_len_np = npy_dict["seq_len"][:train_size]
    print(f"train_np.shape: {train_np.shape}")
    print(f"train_seq_len_np.shape: {train_seq_len_np.shape}")

    valid_np = [npy_dict["input_ids"][train_size:valid_size], npy_dict["labels"][train_size:valid_size],
                npy_dict["attention_mask"][train_size:valid_size], npy_dict["token_type_ids"][train_size:valid_size]]
    valid_np = np.stack(valid_np, axis=-1)
    valid_seq_len_np = npy_dict["seq_len"][train_size:valid_size]
    print(f"valid_np.shape: {valid_np.shape}")
    print(f"valid_seq_len_np.shape: {valid_seq_len_np.shape}")

    test_np = [npy_dict["input_ids"][valid_size:], npy_dict["labels"][valid_size:],
               npy_dict["attention_mask"][valid_size:], npy_dict["token_type_ids"][valid_size:]]
    test_np = np.stack(test_np, axis=-1)
    test_seq_len_np = npy_dict["seq_len"][valid_size:]
    print(f"test_np.shape: {test_np.shape}")
    print(f"test_seq_len_np.shape: {test_seq_len_np.shape}")

    # save
    np.save("../../data/npy/" + mode + "/char_ner/train", train_np)
    np.save("../../data/npy/" + mode + "/char_ner/dev", valid_np)
    np.save("../../data/npy/" + mode + "/char_ner/test", test_np)

    # save seq_len
    np.save("../../data/npy/" + mode + "/char_ner/train_seq_len", train_seq_len_np)
    np.save("../../data/npy/" + mode + "/char_ner/dev_seq_len", valid_seq_len_np)
    np.save("../../data/npy/" + mode + "/char_ner/test_seq_len", test_seq_len_np)

#================================================
# Main
if "__main__" == __name__:
    print(f"[charNER][npy_maker.py] MAIN !")

    is_make_char_npy = True
    if is_make_char_npy:
        src_sent_list = []
        with open("../../data/pkl/old_nikl.pkl", mode="rb") as src_pkl_file:
            src_sent_list = pickle.load(src_pkl_file)
            print(f"[charNER][npy_makery.py] LOAD - {len(src_sent_list)}")

        make_char_npy(mode="old_nikl", tokenizer_name=MODEL_NAME,
                      src_item_list=src_sent_list, max_len=128, use_pos=True)