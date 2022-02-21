import pickle
import os
import numpy as np

from transformers import AutoTokenizer
from data_def import EXO_NE, EXO_NE_Json
from typing import Dict
from dataclasses import field


class EXO_Input_Maker:
    def __init__(self, pkl_path: str="", tokenizer_name: str="klue/roberta-base"):
        print("[EXO_Input_Maker][__init__]----Init")
        if not os.path.exists(pkl_path):
            print(f"f[EXO_Input_Maker][__init__] ERR - Not Exist: {pkl_path}")
            return

        # set path
        self.pkl_path = pkl_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        print("[EXO_Input_Maker][__init__]----End Init")

    def make_input_labels(self):
        print("[EXO_Input_Maker][make_input_labels] Ready...")
        pkl_datasets = []
        with open(self.pkl_path, mode="rb") as pkl_file:
            pkl_datasets = pickle.load(pkl_file)
            print(f"[EXO_Input_Maker][make_input_labels] ne_dataset.len: {len(pkl_datasets)}")

        ne_dict_format = {
            "tokens": [],
            "labels": [],
            "input_ids": [],
            "token_type_ids": [],
            "attention_mask": []
        }

        for src_data in pkl_datasets:
            # text - add [CLS], [SEP]
            text = "[CLS]"
            text += src_data.text
            text += "[SEP]"

            input_ids = self.tokenizer.tokenize(text)
            if 512 < len(input_ids):
                input_ids = input_ids[:511]
                input_ids.append("[SEP]")
            else:
                input_len = len(input_ids)
                input_ids.extend(["[PAD]" for _ in range(512 - input_len)])
            bio_tagging = ["O" for _ in range(len(input_ids))]

            for ne_data in src_data.ne_list:
                cmp_word = ""
                is_ne = False
                begin_idx = 0
                end_idx = 0
                for t_idx, input_token in enumerate(input_ids):
                    if is_ne:
                        cmp_word += input_token.replace("##", "")
                    elif input_token in ne_data.text:
                        begin_idx = t_idx
                        cmp_word = input_token
                        is_ne = True

                    if cmp_word == ne_data.text:
                        ne_tag = ne_data.type.split("_")[0]
                        end_idx = t_idx
                        for ne_idx in range(begin_idx, end_idx+1):
                            if begin_idx == ne_idx:
                                bio_tagging[ne_idx] = "B-"+ne_tag
                            else:
                                bio_tagging[ne_idx] = "I-"+ne_tag
                        begin_idx = end_idx = 0
                        is_ne = False
                        cmp_word = ""
                # end, input_ids
            # end, src_data.ne_list
            ne_dict_format["tokens"].append(input_ids)
            ne_dict_format["labels"].append(bio_tagging)
            ne_dict_format["input_ids"].append(self.tokenizer.convert_tokens_to_ids(input_ids))
            ne_dict_format["token_type_ids"].append([0 for _ in range(512)])
            ne_dict_format["attention_mask"].append([0 for _ in range(512)])
        # end, pkl_datasets

        return ne_dict_format

    def make_npy_files(self, src_dict: Dict[str, list]=field(default_factory=dict),
                       save_path: str=""):
        # token
        tokens = src_dict["tokens"]
        labels = src_dict["labels"]
        input_ids = src_dict["input_ids"]
        token_type_ids = src_dict["token_type_ids"]
        attention_mask = src_dict["attention_mask"]

        tokens = np.array(tokens)
        labels = np.array(labels)
        input_ids = np.array(input_ids)
        token_type_ids = np.array(token_type_ids)
        attention_mask = np.array(attention_mask)
        print(f"[EXO_Input_Maker][make_npy_files] Shape - [token: {tokens.shape}, "
              f"labels: {labels.shape}, input_ids: {input_ids.shape}, "
              f"token_type_ids: {token_type_ids.shape}, attention_mask: {attention_mask.shape}]")

        np.save(save_path + "/tokens", tokens)
        np.save(save_path + "/labels", labels)
        np.save(save_path + "/input_ids", input_ids)
        np.save(save_path + "/token_type_ids", token_type_ids)
        np.save(save_path + "/attention_mask", attention_mask)
        print(f"[EXO_Input_Maker][make_npy_files] Save Complete - {save_path}")

### MAIN ###
if "__main__" == __name__:
    print("[make_input.py][MAIN]")

    exo_maker = EXO_Input_Maker(pkl_path="../datasets/exobrain/res_extract_ne/exo_ne_datasets.pkl",
                                tokenizer_name="monologg/koelectra-base-v3-discriminator")
    res_dict = exo_maker.make_input_labels()
    exo_maker.make_npy_files(src_dict=res_dict, save_path="../datasets/exobrain/npy")