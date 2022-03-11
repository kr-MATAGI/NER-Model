import copy
from typing import List
import pandas as pd


def extract_all_sent_from_naver_ner(src_path: str) -> List:
    ret_list = []

    raw_pd = pd.read_csv(src_path, sep="\t", encoding="utf-8", names=["idx", "word", "tag"])

    sent_buff = []
    for step, item in raw_pd.iterrows():
        if 0 == (step % 10000):
            print(f"{step} Processing...")
        idx = int(item["idx"])
        word = item["word"]
        tag = item["tag"]

        if 0 != step and 1 == idx:
            ret_list.append(copy.deepcopy(" ".join(sent_buff)))
            sent_buff.clear()
            sent_buff.append(word)
        else:
            sent_buff.append(word)
    if 0 < len(sent_buff):
        ret_list.append(copy.deepcopy(sent_buff))

    return ret_list


def check_length_train_dev_naver_ner(train_path: str, dev_path: str) -> List:
    ret_list = []

    with open(train_path, mode="r", encoding="utf-8") as train_file:
        ret_list.extend(copy.deepcopy(train_file.readlines()))

    with open(dev_path, mode="r", encoding="utf-8") as dev_file:
        ret_list.extend(copy.deepcopy(dev_file.readlines()))

    return ret_list


def check_duplicate_data(train_path: str, dev_path: str) -> List:
    train_data = []
    with open(train_path, mode="r", encoding="utf-8") as train_file:
        train_data = copy.deepcopy(train_file.readlines())

    dev_data = []
    with open(dev_path, mode="r", encoding="utf-8") as dev_file:
        dev_data = copy.deepcopy(dev_file.readlines())

    duplicate_count = 0
    for idx, td in enumerate(train_data):
        if 0 == (idx % 10000):
            print(f"{idx} Processing... {td}")
        if td in dev_data:
            duplicate_count += 1
            print(f"{idx} DUP - {td}")
    print(f"duplicate count: {duplicate_count}")


### MAIN ###
if "__main__" == __name__:
    raw_data_train_path = "../../../datasets/Naver_NLP/raw_data_train"
    train_tsv_path = "../../../datasets/Naver_NLP/train.tsv"
    test_tsv_path = "../../../datasets/Naver_NLP/test.tsv"

    is_check_len = False
    if is_check_len:
        # extract - naver ner 'raw' datasets
        raw_ner_list = extract_all_sent_from_naver_ner(src_path=raw_data_train_path)
        print(f"raw ner list len: {len(raw_ner_list)}")

        train_test_list = check_length_train_dev_naver_ner(train_path=train_tsv_path, dev_path=test_tsv_path)
        print(f"train dev len: {len(train_test_list)}")

    is_check_duplicate = True
    if is_check_duplicate:
        check_duplicate_data(train_path=train_tsv_path, dev_path=test_tsv_path)
        # @Note: naver ner 은 '무단 전재 및 재배포 금제' 처럼 중복되는 문장이 들어가 있다. (총 문장 개수 90,000)