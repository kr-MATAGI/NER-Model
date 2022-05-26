import pickle
import numpy as np

from ne_tag_def import ETRI_TAG
from nikl_pos_def import NIKL_POS_TAG

from typing import List
from data_def import Sentence, NE, Morp

from transformers import AutoTokenizer

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


def make_npy(tokenizer_name: str, src_list: List[Sentence], max_len: int=512):
    npy_dict = {
        "input_ids": [],
        "labels": [],
        "attention_mask": [],
        "token_type_ids": [],
        "seq_len": [],
        "pos_tag": [],
    }

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    for proc_idx, sent in enumerate(src_list):
        if 0 == (proc_idx % 1000):
            print(f"{proc_idx} Processing... {sent.text}")
        input_tokens = []
        labels = ["O"] * max_len
        pos_tag_ids = [[0 for _ in range(3)] for _ in range(max_len)]  # [seq_len, 3]

        morp_start_idx = 0
        ne_start_idx = 0
        for word_item in sent.word_list:
            word_tokens = tokenizer.tokenize(word_item.form)

            # Morp
            morp_concat = ""
            for m_idx in range(morp_start_idx, len(sent.morp_list)):
                morp_concat += sent.morp_list[m_idx]
                if

        print(sent.word_list)
        print(sent.morp_list)
        print(sent.ne_list)
        input()


    exit()

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

    split_size = int(len(src_list) * 0.1)
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
    np.save("../data/npy/old_nikl/128/train", train_np)
    np.save("../data/npy/old_nikl/128/dev", dev_np)
    np.save("../data/npy/old_nikl/128/test", test_np)

    # save seq_len
    np.save("../data/npy/old_nikl/128/train_seq_len", train_seq_len_np)
    np.save("../data/npy/old_nikl/128/dev_seq_len", dev_seq_len_np)
    np.save("../data/npy/old_nikl/128/test_seq_len", test_seq_len_np)

    # save pos_tag
    np.save("../data/npy/old_nikl/128/train_pos_tag", train_pos_tag_np)
    np.save("../data/npy/old_nikl/128/dev_pos_tag", dev_pos_tag_np)
    np.save("../data/npy/old_nikl/128/test_pos_tag", test_pos_tag_np)

### MAIN ###
if "__main__" == __name__:
    print("NPY MAKER !")

    all_sent_list = []
    with open("../data/pkl/ne_mp_old_nikl.pkl", mode="rb") as pkl_file:
        all_sent_list = pickle.load(pkl_file)
        print(f"[npy_maker][__main__] all_sent_list size: {len(all_sent_list)}")
    all_sent_list = conv_TTA_ne_category(all_sent_list)
    print(f"main__: all_sent_list.len: {len(all_sent_list)}") # 371,571

    # make npy
    make_npy(tokenizer_name="klue/bert-base", src_list=all_sent_list, max_len=128)