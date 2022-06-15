import numpy
import torch
import torch.nn as nn
import time
import numpy as np
import pandas as pd
import os

from transformers import AutoModel, AutoTokenizer
from bert_custom_model import BERT_LSTM_CRF, BERT_POS_LSTM
from electra_custom_model import ELECTRA_POS_LSTM

from utils.ne_tag_def import ETRI_TAG


def classify_tag_err():
    start_time = time.time()

    model_path = "./model/electra-pos-lstm-crf"
    model = ELECTRA_POS_LSTM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

    target_ids_data = np.load("./target_data/dev.npy")
    target_seq_len_data = np.load("./target_data/dev_seq_len.npy")
    target_pos_tag_data = np.load("./target_data/dev_pos_tag.npy")
    total_data_size = target_ids_data.shape[0]
    print(f"target_ids_data.shape: {target_ids_data.shape}")
    print(f"target_seq_len_data.shape: {target_seq_len_data.shape}")
    print(f"target_pos_tag_data.shape: {target_pos_tag_data.shape}")

    ERROR_COUNT = 0
    labels_list = [x.replace("B-", "").replace("I-", "") for x in ETRI_TAG.keys()]
    ERROR_DICT = {k: [] for k in labels_list}

    model.eval()
    ids_to_tag = {v: k for k, v in ETRI_TAG.items()}
    for data_idx in range(total_data_size):
        if 0 == (data_idx % 100):
            print(f"{data_idx} is Processing... {(data_idx + 1) / total_data_size * 100}")
        inputs = {
            "input_ids": torch.LongTensor([target_ids_data[data_idx, :, 0]]),
            "labels": torch.LongTensor([target_ids_data[data_idx, :, 1]]),
            "attention_mask": torch.LongTensor([target_ids_data[data_idx, :, 2]]),
            "token_type_ids": torch.LongTensor([target_ids_data[data_idx, :, 3]]),
            "input_seq_len": target_seq_len_data[data_idx],
            "pos_tag_ids": torch.LongTensor([target_pos_tag_data[data_idx, :]])
        }

        _, outputs = model(**inputs)
        preds = np.array(outputs)[0]
        #candidate_list = np.argsort(-preds, axis=-1)
        #sort_preds = [ids_to_tag[x] for x in sort_preds]
        #candidate_list = [[ids_to_tag[i] for i in x] for x in candidate_list[0]]
        # preds = np.argmax(preds, axis=2)[0].tolist()

        labels = target_ids_data[data_idx, :, 1]
        text = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        columns = ["text", "label", "preds"]
        rows_list = []
        is_error = False
        err_label_list = []

        for p_idx in range(inputs["input_seq_len"]):
            conv_label = ids_to_tag[labels[p_idx]]
            conv_preds = ids_to_tag[preds[p_idx]]
            #concat_candidate = ", ".join(candidate_list[p_idx][:5])
            if conv_label != conv_preds:
                err_label_list.append(conv_label.replace("B-", "").replace("I-", ""))
                is_error = True
            rows_list.append([text[p_idx], conv_label, conv_preds])
        pd_df = pd.DataFrame(rows_list, columns=columns)

        if is_error:
            ERROR_COUNT += 1
            err_label_list = list(set(err_label_list))
            for err_label in err_label_list:
                err_key = err_label
                ERROR_DICT[err_key].append(pd_df)

            # print(pd_df)
            # break

    # end
    print(f"ERR_COUNT = {ERROR_COUNT}")
    print(f"Total Time = {time.time() - start_time}")

    # write
    for label_key in ERROR_DICT.keys():
        if not os.path.exists("./err_dir/" + label_key):
            os.mkdir("./err_dir/" + label_key)

        for err_idx, err_df in enumerate(ERROR_DICT[label_key]):
            err_df.to_csv("./err_dir/" + label_key + "/" + str(err_idx), sep="\t", encoding="utf-8")

def check_classify_tag_size(path: str):
    '''
        @NOTE
            BERT-base
            {'O': 5296,
            'PS': 139, 'LC': 177,
            'OG': 143, 'AF': 146,
            'DT': 108, 'TI': 31,
            'CV': 506, 'AM': 46,
            'PT': 59, 'QT': 276,
            'FD': 88, 'TR': 24,
            'EV': 41, 'MT': 20,
            'TM': 208}

            BERT-LSTM-CRF
            {'O': 1517,
            'PS': 122, 'LC': 118,
            'OG': 165, 'AF': 128,
            'DT': 101, 'TI': 23,
            'CV': 520, 'AM': 48,
            'PT': 45, 'QT': 261,
            'FD': 78, 'TR': 25,
            'EV': 41, 'MT': 17,
            'TM': 183}

            BERT-LSTM (POS)
            {'O': 1822,
            'PS': 121, 'LC': 130,
            'OG': 141, 'AF': 126,
            'DT': 105, 'TI': 35,
            'CV': 397, 'AM': 39,
            'PT': 55, 'QT': 226,
            'FD': 62, 'TR': 22,
            'EV': 40, 'MT': 19,
            'TM': 228}
    '''
    labels_list = [x.replace("B-", "").replace("I-", "") for x in ETRI_TAG.keys()]
    SIZE_DICT = {k: 0 for k in labels_list}

    list_file = os.listdir(path)
    for tag_name in list_file:
        SIZE_DICT[tag_name] = len(os.listdir(path+"/"+tag_name))
    print(SIZE_DICT)

    sum_val = 0
    for k, v in SIZE_DICT.items():
        sum_val += v
    print(f"total Size: {sum_val}")

def check_LC_OG_country():
    lc_dir = "./err_dir/LC"
    og_dir = "./err_dir/OG"
    LC_file_list = os.listdir(lc_dir)
    OG_file_list = os.listdir(og_dir)

    extract_LC_OG_list = []
    for lc_file in LC_file_list:
        lc_pd_df = pd.read_csv(lc_dir+"/"+lc_file, sep="\t")
        for idx, row in lc_pd_df.iterrows():
            label_tag = row["label"].replace("B-", "").replace("I-", "")
            pred_tag = row["preds"].replace("B-", "").replace("I-", "")
            if ("LC" == label_tag) and ("OG" == pred_tag):
                extract_LC_OG_list.append(lc_pd_df)
                break
    print(f"Extract From LC: {len(extract_LC_OG_list)}")

    for og_file in OG_file_list:
        og_pd_df = pd.read_csv(og_dir+"/"+og_file, sep="\t")
        for idx, row in og_pd_df.iterrows():
            label_tag = row["label"].replace("B-", "").replace("I-", "")
            pred_tag = row["preds"].replace("B-", "").replace("I-", "")
            if ("OG" == label_tag) and ("LC" == pred_tag):
                extract_LC_OG_list.append(og_pd_df)
                break
    print(f"Extract From LC+OG: {len(extract_LC_OG_list)}")

    ## save
    for idx, lc_og_df in enumerate(extract_LC_OG_list):
        lc_og_df.to_csv("./LC_OG/" + str(idx), sep="\t", encoding="utf-8", index=False)
    print("SAVE_LC_OG_Data Frame !")

### MAIN ###
if "__main__" == __name__:
    classify_tag_err()
    #check_classify_tag_size("./err_dir")

    # LC, OG (북한, 중국 나라명이 LC, OG) 오분류 되는 경우
    #check_LC_OG_country()