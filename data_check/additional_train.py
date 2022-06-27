import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from transformers import AutoModel, AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup

from electra_custom_model import ELECTRA_POS_LSTM
from utils.ne_tag_def import ETRI_TAG

### MAIN ###
if "__main__" == __name__:
    RE_labeling_data = [["도 관계자는 “제대로 확인하지 못한 것은 불찰”이라고 말했다.", [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0]],
        ["자기 수요일날 다섯 시 아니 두 시에 당산이 잡혀 있어서 대타를 못 해 줄 것 같다고", [
            0, 0, 9, 10, 11, 12, 0, 0, 0, 25, 26, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0
        ]],
        ["\"첨엔 이 작품이 대체 왜 예술인지 몰랐어요.", [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0
        ]]]

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

    tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    model_path = "./train_model"
    model = ELECTRA_POS_LSTM.from_pretrained(model_path)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-8)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0),
    #                                             num_training_steps=3)
    optimizer.load_state_dict(torch.load(model_path + "/optimizer.pt"))
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to("cuda")
    # scheduler.load_state_dict(torch.load(model_path + "/scheduler.pt"))

    model.to("cuda")
    model.zero_grad()
    model.train()

    total_data_size = total_ids_data.shape[0]
    for re_label_sent in RE_labeling_data:
        re_sent = re_label_sent[0]
        re_label = re_label_sent[1]
        re_tokens = [tokenizer(re_sent, padding='max_length', truncation=True, max_length=128)["input_ids"]]

        for npy_idx in range(total_data_size):
            if 0 == (npy_idx % 10000):
                print(f"{npy_idx} Searching ...")

            inputs = {
                "input_ids": torch.LongTensor([total_ids_data[npy_idx, :, 0]]).to("cuda"),
                "labels": torch.LongTensor([total_ids_data[npy_idx, :, 1]]).to("cuda"),
                "attention_mask": torch.LongTensor([total_ids_data[npy_idx, :, 2]]).to("cuda"),
                "token_type_ids": torch.LongTensor([total_ids_data[npy_idx, :, 3]]).to("cuda"),
                "pos_tag_ids": torch.LongTensor([total_pos_data[npy_idx, :]]).to("cuda")
            }

            if re_tokens != inputs["input_ids"].tolist():
                continue

            print(re_sent)
            print(inputs["labels"])

            # Change label tagging
            inputs["labels"] = torch.LongTensor([re_label]).to("cuda")
            log_likelihood, outputs = model(**inputs)
            loss = -1 * log_likelihood
            loss.backward()

            optimizer.step()
            # scheduler.step()
            model.zero_grad()
            break
    # end, re-train

    print("---------END: RE-TRAIN")

    # eval
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
    ids_to_tag = {v: k for k, v in ETRI_TAG.items()}

    for re_label_sent in RE_labeling_data:
        re_sent = re_label_sent[0]
        re_label = re_label_sent[1]
        re_tokens = [tokenizer(re_sent, padding='max_length', truncation=True, max_length=128)["input_ids"]]

        for npy_idx in range(total_data_size):
            if 0 == (npy_idx % 10000):
                print(f"{npy_idx} Searching ...")

            inputs = {
                "input_ids": torch.LongTensor([total_ids_data[npy_idx, :, 0]]).to("cuda"),
                "labels": torch.LongTensor([total_ids_data[npy_idx, :, 1]]).to("cuda"),
                "attention_mask": torch.LongTensor([total_ids_data[npy_idx, :, 2]]).to("cuda"),
                "token_type_ids": torch.LongTensor([total_ids_data[npy_idx, :, 3]]).to("cuda"),
                "pos_tag_ids": torch.LongTensor([total_pos_data[npy_idx, :]]).to("cuda")
            }

            if re_tokens != inputs["input_ids"].tolist():
                continue

            log_likelihood, outputs = model(**inputs)
            text = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            preds = np.array(outputs)[0]
            labels = total_ids_data[npy_idx, :, 1]

            columns = ["text", "label", "preds"]
            rows_list = []
            for p_idx in range(len(preds)):
                conv_label = ids_to_tag[labels[p_idx]]
                conv_preds = ids_to_tag[preds[p_idx]]
                rows_list.append([text[p_idx], conv_label, conv_preds])
            pd_df = pd.DataFrame(rows_list, columns=columns)

            print("text\tlabel\tpreds")
            for df_idx, df_item in pd_df.iterrows():
                print(df_item["text"], df_item["label"], df_item["preds"])
            print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
            break