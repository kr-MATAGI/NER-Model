import pickle
import os
import numpy as np

from transformers import ElectraTokenizer
from data_def import NE, NE_Json, TTA_NE_tags
from typing import Dict
from dataclasses import field


class Npy_Input_Maker:
    def __init__(self, pkl_path: str="", tokenizer_name: str=""):
        print("[Npy_Input_Maker][__init__]----Init")
        if not os.path.exists(pkl_path):
            print(f"f[Npy_Input_Maker][__init__] ERR - Not Exist: {pkl_path}")
            return

        # set path
        self.pkl_path = pkl_path
        self.tokenizer = ElectraTokenizer.from_pretrained(tokenizer_name)
        print("[Npy_Input_Maker][__init__]----End Init")

    def convert_simple_NE_tag(self, ne_tag: str= ""):
        ret_ne_tag = ne_tag

        if ("LCP" in ret_ne_tag) or ("LCG" in ret_ne_tag): # location
            ret_ne_tag = "LC"
        elif "OGG" in ret_ne_tag: # organization
            ret_ne_tag = "OG"
        elif ("AFW" in ret_ne_tag) or ("AFA" in ret_ne_tag): # artifacts
            ret_ne_tag = "AF"
        elif ("TMM" in ret_ne_tag) or ("TMI" in ret_ne_tag) or \
            ("TMIG" in ret_ne_tag): # term
            ret_ne_tag = "TM"

        return ret_ne_tag

    def make_input_labels(self):
        print("[Npy_Input_Maker][make_input_labels] Ready...")
        pkl_datasets = []
        with open(self.pkl_path, mode="rb") as pkl_file:
            pkl_datasets = pickle.load(pkl_file)
            print(f"[Npy_Input_Maker][make_input_labels] ne_dataset.len: {len(pkl_datasets)}")

        ret_ne_dit = {
            #"tokens": [],
            "labels": [],
            "input_ids": [],
            "token_type_ids": [],
            "attention_mask": []
        }

        '''
            [PAD] = 0
            [UNK] = 1
            [CLS] = 2
            [SEP] = 3
            [MASK] = 4
        '''
        for data_idx, src_data in enumerate(pkl_datasets):
            if 0 == data_idx % 5000: print(f"[Npy_Input_Maker][make_input_labels] {data_idx} Processing...")
            # text - add [CLS], [SEP]
            text = "[CLS]"
            text += src_data.text
            text += "[SEP]"

            token_ids = self.tokenizer.tokenize(text)
            token_ids_len = len(token_ids)
            if 512 < len(token_ids):
                token_ids = token_ids[:511]
                token_ids.append("[SEP]")
                token_ids_len = 512
            else:
                token_len = len(token_ids)
                token_ids.extend(["[PAD]" for _ in range(512 - token_len)])

            # label
            bio_tagging = ["O" for _ in range(token_ids_len)]

            prev_end_idx = 0
            for ne_data in src_data.ne_list:
                ne_tokens = self.tokenizer.tokenize(ne_data.text)

                ne_tokens_len = len(ne_tokens)
                concat_ne_tokens = "".join(ne_tokens)
                for tdx, tok in enumerate(token_ids):
                    if tdx < prev_end_idx:
                        continue

                    if ne_tokens[0] == tok:
                        concat_tokens = "".join(token_ids[tdx:tdx+ne_tokens_len])
                        if concat_tokens == concat_ne_tokens:
                            ne_type = ne_data.type.split("_")[0]
                            for bi_idx in range(tdx, tdx+ne_tokens_len):
                                if bi_idx == tdx:
                                    bio_tagging[bi_idx] = "B-" + self.convert_simple_NE_tag(ne_type)
                                else:
                                    bio_tagging[bi_idx] = "I-" + self.convert_simple_NE_tag(ne_type)
                            prev_end_idx = tdx+ne_tokens_len
                            break
                # end loop, token_ids

            # label
            labels = [TTA_NE_tags["O"] for _ in range(512)]
            for i in range(token_ids_len):
                labels[i] = TTA_NE_tags[bio_tagging[i]]

            # result
            ret_ne_dit["input_ids"].append(self.tokenizer.convert_tokens_to_ids(token_ids))
            ret_ne_dit["labels"].append(labels)
            ret_ne_dit["token_type_ids"].append([0 for _ in range(512)])
            ret_ne_dit["attention_mask"].append([1 if i < token_ids_len else 0 for i in range(512)])

            # end loop, src_data.ne_list
        # end loop, pkl_datasets

        return ret_ne_dit

    def make_npy_files(self, src_dict: Dict[str, list]=field(default_factory=dict),
                       save_path: str=""):
        # token
        #tokens = src_dict["tokens"]
        labels = src_dict["labels"]
        input_ids = src_dict["input_ids"]
        token_type_ids = src_dict["token_type_ids"]
        attention_mask = src_dict["attention_mask"]

        #tokens = np.array(tokens)
        labels = np.array(labels)
        input_ids = np.array(input_ids)
        token_type_ids = np.array(token_type_ids)
        attention_mask = np.array(attention_mask)
        print(f"[Npy_Input_Maker][make_npy_files] Shape - ["
              f"labels: {labels.shape}, input_ids: {input_ids.shape}, "
              f"token_type_ids: {token_type_ids.shape}, attention_mask: {attention_mask.shape}]")

        #np.save(save_path + "/tokens", tokens)
        np.save(save_path + "/labels", labels)
        np.save(save_path + "/input_ids", input_ids)
        np.save(save_path + "/token_type_ids", token_type_ids)
        np.save(save_path + "/attention_mask", attention_mask)
        print(f"[Npy_Input_Maker][make_npy_files] Save Complete - {save_path}")

### MAIN ###
if "__main__" == __name__:
    print("[make_input.py][MAIN]")

    is_make_exo = False
    if is_make_exo:
        exo_maker = Npy_Input_Maker(pkl_path="../datasets/exobrain/res_extract_ne/exo_ne_datasets.pkl",
                                    tokenizer_name="monologg/koelectra-base-v3-discriminator")
        res_dict = exo_maker.make_input_labels()
        exo_maker.make_npy_files(src_dict=res_dict, save_path="../datasets/exobrain/npy")

    is_make_nikl = True
    if is_make_nikl:
        ## Split npy
        # train: 939,701 // filtered: 389,505
        # eval:  134,243 // filtered: 55,643
        # test:  268,487 // filtered: 111,289
        is_split_pkl = True
        if is_split_pkl:
            with open("../datasets/NIKL/res_nikl_ne/mecab_jk_nikl_datasets.pkl", mode="rb") as origin_file:
                origin_list = pickle.load(origin_file)
                train_size = int(len(origin_list) * 0.7)
                eval_size = int(len(origin_list) * 0.1)
                train_data_list = origin_list[:train_size]
                eval_data_list = origin_list[train_size:train_size+eval_size]
                test_data_list = origin_list[train_size+eval_size:]
                print(f"train_size: {len(train_data_list)}, eval_size: {len(eval_data_list)}, "
                      f"test_size: {len(test_data_list)}")

                with open("../datasets/NIKL/res_nikl_ne/mecab_jk_nikl_ne_train.pkl", mode="wb") as train_file:
                    pickle.dump(train_data_list, train_file)
                with open("../datasets/NIKL/res_nikl_ne/mecab_jk_nikl_ne_eval.pkl", mode="wb") as eval_file:
                    pickle.dump(eval_data_list, eval_file)
                with open("../datasets/NIKL/res_nikl_ne/mecab_jk_nikl_ne_test.pkl", mode="wb") as test_file:
                    pickle.dump(test_data_list, test_file)

        # make npy
        train_maker = Npy_Input_Maker(pkl_path="../datasets/NIKL/res_nikl_ne/mecab_jk_nikl_ne_train.pkl",
                                      tokenizer_name="monologg/koelectra-base-v3-discriminator")
        train_dict = train_maker.make_input_labels()
        train_maker.make_npy_files(src_dict=train_dict, save_path="../datasets/NIKL/npy/mecab/train")

        eval_maker = Npy_Input_Maker(pkl_path="../datasets/NIKL/res_nikl_ne/mecab_jk_nikl_ne_eval.pkl",
                                     tokenizer_name="monologg/koelectra-base-v3-discriminator")
        eval_dict = eval_maker.make_input_labels()
        eval_maker.make_npy_files(src_dict=eval_dict, save_path="../datasets/NIKL/npy/mecab/eval")

        test_maker = Npy_Input_Maker(pkl_path="../datasets/NIKL/res_nikl_ne/mecab_jk_nikl_ne_test.pkl",
                                     tokenizer_name="monologg/koelectra-base-v3-discriminator")
        test_dict = test_maker.make_input_labels()
        test_maker.make_npy_files(src_dict=test_dict, save_path="../datasets/NIKL/npy/mecab/test")