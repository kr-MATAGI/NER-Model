import numpy as np

import sys
sys.path.append("C:/Users/MATAGI/Desktop/Git/De-identification-with-NER/Utils")

from transformers import ElectraTokenizer
from Utils.datasets_maker.nikl.data_def import TTA_NE_tags
from Utils.datasets_maker.naver.naver_def import NAVER_NE_MAP

### MAIN ###
if "__main__" == __name__:
    tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

    #target_path = "../../datasets/NIKL/npy/mecab/train"
    target_path = "../../datasets/Naver_NLP/npy/train"

    input_ids_path = target_path+"/input_ids.npy"
    labels_path = target_path+"/labels.npy"
    attention_mask_path = target_path+"/attention_mask.npy"
    token_type_ids_path = target_path+"/token_type_ids.npy"

    # load npy
    input_ids = np.load(input_ids_path)
    labels = np.load(labels_path)
    attention_mask = np.load(attention_mask_path)
    token_type_ids = np.load(token_type_ids_path)

    print(f"input_ids.shape: {input_ids.shape}, labels.shape: {labels.shape}")

    list_len = input_ids.shape[-1] # 512
    #id2label = {int(val): key for key, val in TTA_NE_tags.items()} # NIKL
    id2label = {int(val): key for key, val in NAVER_NE_MAP.items()}  # NIKL
    for input_id, label, at_mask, tt_ids in zip(input_ids, labels, attention_mask, token_type_ids):
        print(f"input_id.len: {len(input_id)}, label.len: {len(label)}, "
              f"at_mask.len: {len(at_mask)}, tt_ids.len: {len(tt_ids)}")
        token_list = tokenizer.convert_ids_to_tokens(input_id)

        for idx in range(list_len):
            try:
                print(f"{token_list[idx]} : {id2label[label[idx]]}-{label[idx]} : {at_mask[idx]} : {tt_ids[idx]}")
                if ("[PAD]" == token_list[idx]): break
            except:
                print(f"{token_list[idx]} : {label[idx]} : {at_mask[idx]} : {tt_ids[idx]}")
                if("[PAD]" == token_list[idx]): break

        input()


