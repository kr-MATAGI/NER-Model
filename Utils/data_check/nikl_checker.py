import pickle
import matplotlib.pyplot as plt
import numpy as np

from typing import Dict

import platform
import sys

if "Darwin" == platform.system():
    sys.path.append("/Users/matagi/Desktop/Git/De-identification-with-NER/Utils")
else:
    sys.path.append("C:/Users/MATAGI/Desktop/Git/De-identification-with-NER/Utils")

from Utils.data_def import TTA_NE_tags

def convert_NE_tag(ne_tag: str = ""):
    ret_ne_tag = ne_tag

    if ("LCP" in ret_ne_tag) or ("LCG" in ret_ne_tag):  # location
        ret_ne_tag = "LC"
    elif "OGG" in ret_ne_tag:  # organization
        ret_ne_tag = "OG"
    elif ("AFW" in ret_ne_tag) or ("AFA" in ret_ne_tag):  # artifacts
        ret_ne_tag = "AF"
    elif ("TMM" in ret_ne_tag) or ("TMI" in ret_ne_tag) or \
            ("TMIG" in ret_ne_tag):  # term
        ret_ne_tag = "TM"

    return ret_ne_tag

def check_NIKL_tag_count(data_path: str) -> None:
    # count dict

    # count dict
    count_dict = {
        "PS": 0, "LC": 0, "OG": 0, "AF": 0, "DT": 0, "TI": 0,
        "CV": 0, "AM": 0, "PT": 0, "QT": 0, "FD": 0, "TR": 0,
        "EV": 0, "MT": 0, "TM": 0
    }
    print(f"tag size: {len(count_dict.keys())}")

    with open(data_path, mode="rb") as data_file:
        ne_data_list = pickle.load(data_file)

        for ne_json in ne_data_list:
            for ne in ne_json.ne_list:
                ne_tag = convert_NE_tag(ne.type.split("_")[0])
                count_dict[ne_tag] += 1

    print(f"result: \n{sorted(count_dict.items(), key=lambda item: item[1], reverse=True),}")

    # plot
    tag_name = count_dict.keys()
    tag_count = [count_dict[k] for k in count_dict.keys()]

    x = np.arange(15)
    plt.bar(x, tag_count)
    plt.xlabel("tag name")
    plt.ylabel("count")
    plt.xticks(x, tag_name)
    plt.show()

def check_NIKL_sent_len(data_path: str) -> None:
    len_dict = {}

    text_count = 0
    max_len = 0
    min_len = 9999
    sum_len = 0
    with open(data_path, mode="rb") as data_file:
        ne_data_list = pickle.load(data_file)

        text_count = len(ne_data_list)
        for ne_json in ne_data_list:
            text_len = len(ne_json.text)
            sum_len += text_len

            max_len = text_len if max_len < text_len else max_len
            min_len = text_len if min_len > text_len else min_len

            key = text_len // 10
            if key  not in len_dict.keys():
                len_dict[key] = 1
            else:
                len_dict[key] += 1

    print(f"total text count: {text_count}")
    print(f"max len: {max_len}")
    print(f"min len: {min_len}")
    print(f"avg len: {sum_len/text_count:.4f}")
    print(f"dict: {len_dict}")

    # plot
    xtick_keys = [k for k in len_dict.keys()]

    x = np.arange(len(len_dict.keys()))
    plt.bar(x, len_dict)
    plt.xlabel("text len.key")
    plt.ylabel("value")
    plt.xticks(x, xtick_keys)
    plt.show()

def check_NIKL_sent_len_filter(data_path: str) -> None:
    filter_count = 0
    origin_size = 0
    with open(data_path, mode="rb") as data_file:
        ne_data_list = pickle.load(data_file)

        origin_size = len(ne_data_list)
        for ne_json in ne_data_list:
            if 10 > len(ne_json.text):
                filter_count += 1

    print(f"forigin size: {origin_size}")
    print(f"filter count: {filter_count}")
    print(f"diff size: {origin_size - filter_count}")

def check_NIKL_ne_tag_empty(data_path: str) -> None:
    ne_empty_size = 0

    with open(data_path, mode="rb") as data_file:
        ne_data_list = pickle.load(data_file)

        for ne_json in ne_data_list:
            if 0 >= len(ne_json.ne_list):
                ne_empty_size += 1

    print(f"NE Tag is Empty: {ne_empty_size}")

def check_NIKL_ne_empty_sent_len(data_path: str) -> None:
    filter_sent_len = 0
    filter_ne_empty_size = 0
    filter_all_cond_size = 0

    total_size = 0
    with open(data_path, mode="rb") as data_file:
        ne_data_list = pickle.load(data_file)

        total_size = len(ne_data_list)
        for ne_json in ne_data_list:
            is_ne_empty = False
            if 0 >= len(ne_json.ne_list):
                filter_ne_empty_size += 1
                is_ne_empty = True

            is_filter_sent_len = False
            if 10 > len(ne_json.text):
                filter_sent_len += 1
                is_filter_sent_len = True

            if any([is_ne_empty, is_filter_sent_len]):
                filter_all_cond_size += 1

    print(f"ne_empty: {filter_ne_empty_size}, diff: {total_size - filter_ne_empty_size}")
    print(f"sent_len: {filter_sent_len}, diff: {total_size - filter_sent_len}")
    print(f"all_cond: {filter_all_cond_size}, diff: {total_size - filter_all_cond_size}")


### MAIN ###
if "__main__" == __name__:
    nikl_path = "../../datasets/NIKL/res_nikl_ne/nikl_ne_datasets.pkl"

    is_check_nikl_tag_count = False
    if is_check_nikl_tag_count:
        check_NIKL_tag_count(data_path=nikl_path)


    is_check_nikl_sent_len = False
    if is_check_nikl_sent_len:
        check_NIKL_sent_len(data_path=nikl_path)

    is_check_nikl_sent_len_filter = False
    if is_check_nikl_sent_len_filter:
        check_NIKL_sent_len_filter(data_path=nikl_path)

    is_check_nikl_ne_empty = False
    if is_check_nikl_ne_empty:
        check_NIKL_ne_tag_empty(data_path=nikl_path)

    is_check_nikl_filter = True
    if is_check_nikl_filter:
        check_NIKL_ne_empty_sent_len(data_path=nikl_path)