import torch
import torch.nn as nn
import time
import numpy as np
import pandas as pd
import os

from transformers import AutoModel, AutoTokenizer
from bert_custom_model import BERT_LSTM_CRF

from utils.ne_tag_def import ETRI_TAG


### MAIN ###
if "__main__" == __name__:
    start_time = time.time()

    model_path = "./model/klue-bert-base-lstm-crf"
    model = BERT_LSTM_CRF.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

    target_ids_data = np.load("./target_data/dev.npy")
    target_seq_len_data = np.load("./target_data/dev_seq_len.npy")
    total_data_size = target_ids_data.shape[0]
    print(f"target_ids_data.shape: {target_ids_data.shape}")
    print(f"target_seq_len_data.shape: {target_seq_len_data.shape}")

    ERROR_COUNT = 0
    labels_list = [x.replace("B-", "").replace("I-", "") for x in ETRI_TAG.keys()]
    ERROR_DICT = {k:[] for k in labels_list}

    model.eval()
    ids_to_tag = {v:k for k, v in ETRI_TAG.items()}
    for data_idx in range(total_data_size):
        if 0 == (data_idx % 100):
            print(f"{data_idx} is Processing... {(data_idx+1)/total_data_size * 100}")
        inputs = {
            "input_ids": torch.LongTensor([target_ids_data[data_idx, :, 0]]),
            # "labels": torch.LongTensor([target_ids_data[data_idx, :, 1]]),
            "attention_mask": torch.LongTensor([target_ids_data[data_idx, :, 2]]),
            "token_type_ids": torch.LongTensor([target_ids_data[data_idx, :, 3]]),
            # "seq_len": [target_seq_len_data[data_idx]]
        }

        outputs = model(**inputs)
        preds = outputs[0]
        labels = target_ids_data[data_idx, :, 1]
        text = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        columns = ["text", "label", "preds"]
        rows_list = []
        is_error = False
        err_label_list = []
        for p_idx in range(len(preds)):
            conv_label = ids_to_tag[labels[p_idx]]
            conv_preds = ids_to_tag[preds[p_idx]]
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
            #print(pd_df)

    # end
    print(f"ERR_COUNT = {ERROR_COUNT}")
    print(f"Total Time = {time.time()-start_time}")

    # write
    for label_key in ERROR_DICT.keys():
        if not os.path.exists("./err_dir/"+label_key):
            os.mkdir("./err_dir/"+label_key)

        for err_idx, err_df in enumerate(ERROR_DICT[label_key]):
            err_df.to_csv("./err_dir/"+label_key+"/"+str(err_idx), sep="\t", encoding="utf-8")