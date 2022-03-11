from naver_def import NAVER_NE
from typing import List

import pickle
import copy

from transformers import ElectraTokenizer

def extractor_naver_ner_sent(raw_file_path: str) -> List:
    ret_list = []

    with open(raw_file_path, mode="r", encoding="utf-8") as raw_file:
        raw_file_lines = raw_file.readlines()

        naver_ne = NAVER_NE()
        for idx, line in enumerate(raw_file_lines):
            if 0 == (idx % 10000):
                print(f"{idx} Processing...")
            if 1 == len(line):
                ret_list.append(copy.deepcopy(naver_ne))
                naver_ne = NAVER_NE()
            else:
                line_sp = line.split("\t")
                naver_ne.sent_list.append(line_sp[1])
                naver_ne.tag_list.append(line_sp[-1].replace("\n", ""))

    print(f"result.len: {len(ret_list)}")
    return ret_list

def make_npy(src_list: List[NAVER_NE], save_path: str, mode: str, model_name: str, max_len: int = 512) -> None:
    print(f"mode: {mode}, src_list.len: {len(src_list)}")

    input_ids = []
    label_ids = []
    token_type_ids = []
    attention_mask = []
    tokenizer = ElectraTokenizer.from_pretrained(model_name)
    for src_data in src_list:
        sentence = " ".join(src_data.sent_list)
        tokens = tokenizer.tokenize(sentence)

        print(sentence)
        print(tokenizer.tokenize(sentence))
        print(src_data.sent_list)
        print(src_data.tag_list)

        break


### MAIN ###
if "__main__" == __name__:
    is_extract_naver = False
    if is_extract_naver:
        file_path = "../../../datasets/Naver_NLP/raw_data_train"
        ret_list = extractor_naver_ner_sent(file_path)
        with open("../../../datasets/Naver_NLP/raw_data.pkl", mode="wb") as save_pkl:
            pickle.dump(ret_list, save_pkl)

    is_make_npy = True
    if is_make_npy:
        load_list = []
        with open("../../../datasets/Naver_NLP/raw_data.pkl", mode="rb") as load_pkl:
            load_list = pickle.load(load_pkl)
        total_size = len(load_list)
        train_list = load_list[:int(total_size*0.9)]
        dev_list = load_list[int(total_size*0.9):]
        print(f"train.len: {len(train_list)}, dev.len: {len(dev_list)}")

        make_npy(src_list=train_list, save_path="../../../datasets/Naver_NLP/npy/train", mode="train",
                 model_name="monologg/koelectra-base-v3-discriminator")
        make_npy(src_list=dev_list, save_path="../../../datasets/Naver_NLP/npy/test", mode="test",
                 model_name="monologg/koelectra-base-v3-discriminator")

