import pickle
import numpy as np
import copy
from typing import List

import sys
sys.path.append("C:/Users/MATAGI/Desktop/Git/De-identification-with-NER/Utils/datasets_maker/naver")
from Utils.datasets_maker.naver.naver_def import NAVER_NE, NAVER_NE_MAP

from transformers import AutoTokenizer

def make_npy(src_list: List[NAVER_NE], save_path: str, mode: str, model_name: str, max_len: int = 512) -> None:
    print(f"mode: {mode}, src_list.len: {len(src_list)}")

    results = {
        "input_ids": [],
        "labels": [],
        "attention_mask": [],
        "token_type_ids": []
    }

    '''
        [PAD] = 0
        [UNK] = 1
        [CLS] = 2
        [SEP] = 3
        [MASK] = 4
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for src_idx, src_data in enumerate(src_list):
        if 0 == (src_idx % 1000):
            print(f"{src_idx} Processing...")

        tokens = tokenizer.tokenize(src_data.sentence)
        tokens.insert(0, "[CLS]")
        tokens.append("[SEP]")
        tokens_len = len(tokens)

        if max_len < tokens_len:
            tokens = tokens[:(max_len - 1)]
            tokens.append("[SEP]")
            tokens_len = max_len
        else:
            tokens.extend(["[PAD]" for _ in range(max_len - tokens_len)])

        # convert tagging
        bio_tagging = ["O" for _ in range(tokens_len)]
        bio_tagging[0] = bio_tagging[-1] = -100  # not exists dict.keys()

        tag_list_len = len(src_data.tag_list)
        prev_token_end_idx = 0

        for idx in range(tag_list_len):
            target_word_tokens = tokenizer.tokenize(src_data.word_list[idx])
            if 0 >= len(target_word_tokens):
                continue
            target_ne = src_data.tag_list[idx].split("_")[0]
            BI_type = src_data.tag_list[idx].split("_")[-1]

            for tk_idx, tok in enumerate(tokens):
                if (0 == tk_idx) or ((tokens_len - 1) == tk_idx):  # ignore [CLS], [SEP]
                    continue

                if prev_token_end_idx >= tk_idx:
                    continue

                if target_word_tokens[0] == tok:
                    tg_word_tokens_len = len(target_word_tokens)
                    concat_tokens = tokens[tk_idx:(tk_idx + tg_word_tokens_len)]

                    if concat_tokens == target_word_tokens:
                        prev_token_end_idx = tk_idx
                        for bio_idx in range(tk_idx, (tk_idx + tg_word_tokens_len)):
                            if "-" == target_ne and bio_idx == tk_idx:
                                bio_tagging[bio_idx] = "O"
                            elif "-" != target_ne and bio_idx == tk_idx:
                                bio_tagging[bio_idx] = BI_type + "-" + target_ne
                            else:
                                bio_tagging[bio_idx] = -100
                        break
            # end loop, tokens
        # end loop, range(tag_list_len)

        # make input_ids, labels, attention mask, token_type_ids
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        bio_tagging.extend(-100 for _ in range(max_len - tokens_len))
        labels = [NAVER_NE_MAP[tag] if tag in NAVER_NE_MAP.keys() else -100 for tag in bio_tagging]
        attention_mask = [1 if i < tokens_len else 0 for i in range(max_len)]
        token_type_ids = [0 for _ in range(max_len)]

        results["input_ids"].append(copy.deepcopy(input_ids))
        results["labels"].append(copy.deepcopy(labels))
        results["attention_mask"].append(copy.deepcopy(attention_mask))
        results["token_type_ids"].append(copy.deepcopy(token_type_ids))
    # end loop, src_list

    # save file
    results["input_ids"] = np.array(results["input_ids"])
    results["labels"] = np.array(results["labels"])
    results["attention_mask"] = np.array(results["attention_mask"])
    results["token_type_ids"] = np.array(results["token_type_ids"])
    print(f"input_ids.shape: {results['input_ids'].shape}")
    print(f"labels.shape: {results['labels'].shape}")
    print(f"attention_mask.shape: {results['attention_mask'].shape}")
    print(f"token_type_ids.shape: {results['token_type_ids'].shape}")

    np.save(save_path + "/input_ids", results["input_ids"])
    np.save(save_path + "/labels", results["labels"])
    np.save(save_path + "/attention_mask", results["attention_mask"])
    np.save(save_path + "/token_type_ids", results["token_type_ids"])

if "__main__" == __name__:
    is_make_klue_naver_ner = True
    if is_make_klue_naver_ner:
        load_list = []
        with open("../datasets/Naver_NLP/raw_data.pkl", mode="rb") as load_pkl:
            load_list = pickle.load(load_pkl)

        total_size = len(load_list)
        train_list = load_list[:int(total_size * 0.9)]
        dev_list = load_list[int(total_size * 0.9):]
        print(f"train.len: {len(train_list)}, dev.len: {len(dev_list)}")

        make_npy(src_list=train_list, save_path="./npy/train", mode="train",
                 model_name="klue/bert-base", max_len=512)
        make_npy(src_list=dev_list, save_path="./npy/test", mode="test",
                 model_name="klue/bert-base", max_len=512)