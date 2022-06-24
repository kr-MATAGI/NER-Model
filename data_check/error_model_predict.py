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

### METHOD ###
def error_predict_by_model(sent: str="", tokenizer_name: str="", model_path: str=""):
    # load model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = ELECTRA_POS_LSTM.from_pretrained(model_path)

    # load data
    train_ids_data = np.load("./target_data/train.npy")
    train_pos_data = np.load("./target_data/train_pos_tag.npy")
    dev_ids_data = np.load("./target_data/dev.npy")
    dev_pos_data = np.load("./target_data/dev_pos_tag.npy")
    test_ids_data = np.load("./target_data/test.npy")
    test_pos_data = np.load("./target_data/test_pos_tag.npy")
    total_ids_data = np.vstack([train_ids_data, dev_ids_data, test_ids_data])
    total_pos_data = np.vstack([train_pos_data, dev_pos_data, test_pos_data])
    print(f"train_ids_data.shape: {train_ids_data.shape}")
    print(f"train_pos_data.shape: {train_pos_data.shape}")
    print(f"dev_ids_data.shape: {dev_ids_data.shape}")
    print(f"dev_pos_data.shape: {dev_pos_data.shape}")
    print(f"test_ids_data.shape: {test_ids_data.shape}")
    print(f"test_pos_data.shape: {test_pos_data.shape}")
    print(f"total_ids_data.shape: {total_ids_data.shape}")
    print(f"total_pos_data.shape: {total_pos_data.shape}")

    model.eval()
    total_data_size = total_ids_data.shape[0]
    ids_to_tag = {v: k for k, v in ETRI_TAG.items()}
    target_tokens = [tokenizer(sent, padding='max_length', truncation=True, max_length=128)["input_ids"]]
    for data_idx in range(total_data_size):
        if 0 == (data_idx % 10000):
            print(f"[error_model_predict][error_predict_by_model] {data_idx} is processing...")

        inputs = {
            "input_ids": torch.LongTensor([total_ids_data[data_idx, :, 0]]),
            "labels": torch.LongTensor([total_ids_data[data_idx, :, 1]]),
            "attention_mask": torch.LongTensor([total_ids_data[data_idx, :, 2]]),
            "token_type_ids": torch.LongTensor([total_ids_data[data_idx, :, 3]]),
            "pos_tag_ids": torch.LongTensor([total_pos_data[data_idx, :]])
        }

        if target_tokens != inputs["input_ids"].tolist():
            continue
        else:
            print("FIND !")

        log_likelihood, outputs = model(**inputs)

        text = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        preds = np.array(outputs)[0]
        labels = total_ids_data[data_idx, :, 1]

        columns = ["text", "label", "preds"]
        rows_list = []
        for p_idx in range(len(preds)):
            conv_label = ids_to_tag[labels[p_idx]]
            conv_preds = ids_to_tag[preds[p_idx]]
            rows_list.append([text[p_idx], conv_label, conv_preds])
        pd_df = pd.DataFrame(rows_list, columns=columns)

        print("text\tlabel\tpreds" )
        for df_idx, df_item in pd_df.iterrows():
            print(df_item["text"], df_item["label"], df_item["preds"])
        return

### MAIN ###
if "__main__" == __name__:
    print("[error_model_predict][__main__] __MAIN__")

    target_sent = "현재 서울 지하철 1~9호선, 분당선 등 299개 역 승강장 스크린 도어 4840칸에 시가 걸려 있는데, 되레 시와 시민을 멀어지게 하고 있다는 비판이다."
    tokenizer_name = "monologg/koelectra-base-v3-discriminator"
    model_path = "./model/electra-pos-lstm-crf"
    error_predict_by_model(sent=target_sent, tokenizer_name=tokenizer_name, model_path=model_path)