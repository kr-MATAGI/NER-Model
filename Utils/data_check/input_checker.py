import numpy as np

import sys
sys.path.append("C:/Users/MATAGI/Desktop/Git/De-identification-with-NER/Utils")

from transformers import ElectraTokenizer
from Utils.data_def import TTA_NE_tags

### MAIN ###
if "__main__" == __name__:
    tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

    target_path = "../../datasets/exobrain/npy/ko-electra-base"
    input_ids_path = target_path+"/input_ids.npy"
    labels_path = target_path+"/labels.npy"

    # load npy
    input_ids = np.load(input_ids_path)
    labels = np.load(labels_path)

    print(f"input_ids.shape: {input_ids.shape}, labels.shape: {labels.shape}")

    list_len = input_ids.shape[-1] # 512
    id2label = {int(val): key for key, val in TTA_NE_tags.items()}
    for input_id, label in zip(input_ids, labels):
        print(f"input_id.len: {len(input_id)}, label.len: {len(label)}")
        token_list = tokenizer.convert_ids_to_tokens(input_id)

        for idx in range(list_len):
            print(f"{token_list[idx]} : {id2label[label[idx]]}")

        input()

