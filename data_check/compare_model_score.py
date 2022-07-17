import os

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from transformers import AutoTokenizer

from electra_custom_model import ELECTRA_POS_LSTM
from utils.ne_tag_def import ETRI_TAG


#====================================================
def compare_models_output():
#====================================================
    # INIT
    g_err_cnt = 0

    # data
    dev_ids_data = np.load("./target_data/dev.npy")
    dev_pos_data = np.load("./target_data/dev_pos_tag.npy")
    print(f"dev_ids_data.shape: {dev_ids_data.shape}")
    print(f"dev_pos_data.shape: {dev_pos_data.shape}")

    # model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path_1 = "./model/electra-pos-lstm-crf"
    model_1 = ELECTRA_POS_LSTM.from_pretrained(model_path_1)
    model_1.to(device)
    model_1.eval()

    model_path_2 = "./train_model"
    model_2 = ELECTRA_POS_LSTM.from_pretrained(model_path_2)
    model_2.to(device)
    model_2.eval()

    # target
    dev_data_size = dev_ids_data.shape[0]
    target_ne = ["PS", "OG", "AF", "TI", "CV",
                 "PT", "QT", "FD", "TR", "AM",
                 "DT", "LC", "MT", "TM", "EV"]
    target_bio_ne = []
    target_bio_ne.extend(["B-"+ne for ne in target_ne])
    target_bio_ne.extend(["I-"+ne for ne in target_ne])
    print(f"target_bio_ne: {target_bio_ne}")

    # result dict
    g_error_dict = {k: [] for k in target_ne}

    tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    for data_idx in range(dev_data_size):
        if 0 == (data_idx % 1000):
            print(f"{data_idx} is processing...")

        inputs = {
            "input_ids": torch.LongTensor([dev_ids_data[data_idx, :, 0]]).to(device),
            "labels": torch.LongTensor([dev_ids_data[data_idx, :, 1]]).to(device),
            "attention_mask": torch.LongTensor([dev_ids_data[data_idx, :, 2]]).to(device),
            "token_type_ids": torch.LongTensor([dev_ids_data[data_idx, :, 3]]).to(device),
            "pos_tag_ids": torch.LongTensor([dev_pos_data[data_idx, :]]).to(device)
        }

        conv_text = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        log_liklihood_1, outputs_1 = model_1(**inputs)
        log_liklihood_2, outputs_2 = model_2(**inputs)

        preds_1 = np.array(outputs_1)[0].tolist()
        preds_2 = np.array(outputs_2)[0].tolist()
        labels = dev_ids_data[data_idx, :, 1]

        ids_to_tag = {v: k for k, v in ETRI_TAG.items()}
        columns = ["text", "label", "preds_1", "preds_2"]
        rows_list = []
        conv_preds_list_1 = []
        conv_preds_list_2 = []
        is_error = False
        err_key = ""
        for p_idx in range(len(preds_1)):
            conv_label = ids_to_tag[labels[p_idx]]
            conv_preds_1 = ids_to_tag[preds_1[p_idx]]
            conv_preds_2 = ids_to_tag[preds_2[p_idx]]
            # if (conv_label != conv_preds_1) or (conv_label != conv_preds_2):
            if conv_label != conv_preds_2:
                is_error = True
                if 0 >= len(err_key):
                    err_key = conv_label
            conv_preds_list_1.append(conv_preds_1)
            conv_preds_list_2.append(conv_preds_2)
            rows_list.append([conv_text[p_idx], conv_label, conv_preds_1, conv_preds_2])
        pd_df = pd.DataFrame(rows_list, columns=columns)

        if preds_1 != preds_2:
            filter_ne_1 = list(filter(lambda x: x in conv_preds_list_1, target_bio_ne))
            filter_ne_2 = list(filter(lambda x: x in conv_preds_list_2, target_bio_ne))

            # label != preds
            if is_error and (0 < len(filter_ne_1) or 0 < len(filter_ne_2)):
                g_err_cnt += 1
                if 0 < len(filter_ne_1):
                    g_error_dict[filter_ne_1[0].split("-")[-1]].append(pd_df)
                else:
                    g_error_dict[filter_ne_2[0].split("-")[-1]].append(pd_df)

    # write
    cmp_dir_path = "./cmp_dir/"
    for label_key in g_error_dict.keys():
        if not os.path.exists(cmp_dir_path + label_key):
            os.mkdir(cmp_dir_path + label_key)

        for err_idx, err_df in enumerate(g_error_dict[label_key]):
            err_df.to_csv(cmp_dir_path+label_key+"/"+str(err_idx), sep="\t", encoding="utf-8", index=False)

#====================================================
def check_incorrect_outputs(target_dir: str, target_ne: str):
#====================================================
    tsv_file_list = os.listdir(target_dir)
    print(f"[compare_model_score][check_incorrect_outputs] tsv size: {len(tsv_file_list)}")

    incorrect_count_1: int = 0 # model_1 (preds_1)
    incorrect_count_2: int = 0 # model_2 (preds_2)
    diff_count: int = 0 # preds_1 != preds_2

    for tsv_file_name in tsv_file_list:
        full_path = target_dir+"/"+tsv_file_name
        tsv_file_contents = pd.read_csv(full_path, sep="\t", encoding="utf-8")

        for idx, row in tsv_file_contents.iterrows():
            label = row["label"]
            preds_1 = row["preds_1"]
            preds_2 = row["preds_2"]

            if target_ne in label and "B-" in label:
                if label != preds_1:
                    incorrect_count_1 += 1
                if label != preds_2:
                    incorrect_count_2 += 1
                if preds_1 != preds_2:
                    diff_count += 1
    print(f"[compare_model_score][check_incorrect_outputs] incorrect_count_1: {incorrect_count_1}")
    print(f"[compare_model_score][check_incorrect_outputs] incorrect_count_2: {incorrect_count_2}")
    print(f"[compare_model_score][check_incorrect_outputs] diff_count: {diff_count}")

### MAIN ###
if "__main__" == __name__:
    is_write_compare_outputs = False
    if is_write_compare_outputs:
        compare_models_output()

    '''
        @NOTE
            1. PS
                - tsv size: 1115
                - incorrect_count_1: 1523
                - incorrect_count_2: 48
                - diff_count: 1509
            
            2. LC
                - tsv size: 98
                - incorrect_count_1: 107
                - incorrect_count_2: 6
                - diff_count: 104
            
            3. OG
                - tsv size: 1074
                - incorrect_count_1: 1398
                - incorrect_count_2: 185
                - diff_count: 1346
            
            4. AF
                - tsv size: 316
                - incorrect_count_1: 295
                - incorrect_count_2: 41
                - diff_count: 269
            
            5. DT
                - tsv size: 144
                - incorrect_count_1: 167
                - incorrect_count_2: 16
                - diff_count: 164
             
            6. TI
                - tsv size: 60
                - incorrect_count_1: 57
                - incorrect_count_2: 6
                - diff_count: 56
                
            7. CV
                - tsv size: 823
                - incorrect_count_1: 685
                - incorrect_count_2: 125
                - diff_count: 622
            
            8. AM
                - tsv size: 83
                - incorrect_count_1: 52
                - incorrect_count_2: 4
                - diff_count: 50
            
            9. PT
                - tsv size: 23
                - incorrect_count_1: 11
                - incorrect_count_2: 3
                - diff_count: 10
                
            10. QT
                - tsv size: 392
                - incorrect_count_1: 520
                - incorrect_count_2: 66
                - diff_count: 501
            
            11. FD
                - tsv size: 1234
                - incorrect_count_1: 54
                - incorrect_count_2: 23
                - diff_count: 50
            
            12. TR
                - tsv size: 27
                - incorrect_count_1: 9
                - incorrect_count_2: 2
                - diff_count: 8
            
            13. EV
                - tsv size: 49
                - incorrect_count_1: 30
                - incorrect_count_2: 8
                - diff_count: 27
            
            14. MT
                - tsv size: 19
                - incorrect_count_1: 15
                - incorrect_count_2: 1
                - diff_count: 14
            
            15. TM
                - tsv size: 106
                - incorrect_count_1: 86
                - incorrect_count_2: 23
                - diff_count: 70
    '''
    target_ne = "TM"
    target_dir_path = "./cmp_dir/" + target_ne
    check_incorrect_outputs(target_dir=target_dir_path, target_ne=target_ne)
