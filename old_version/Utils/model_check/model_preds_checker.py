import sys
sys.path.append("C:/Users/MATAGI/Desktop/Git/De-identification-with-NER/Utils")

import numpy as np
import torch
from transformers import ElectraTokenizer
from Utils.datasets_maker.nikl.data_def import TTA_NE_tags

def make_input_by_sentence(tokenizer, input_str: str=""):
    ret_dict = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": []
    }

    '''
        [PAD] = 0
        [UNK] = 1
        [CLS] = 2
        [SEP] = 3
        [MASK] = 4
    '''

    text = "[CLS]"
    text += input_str
    text += "[SEP]"

    # attention_mask
    attention_size = len(text)

    # token_ids
    token_ids = tokenizer.tokenize(text)
    if 512 < len(token_ids):
        token_ids = token_ids[:511]
        token_ids.append("[SEP]")
    else:
        token_len = len(token_ids)
        token_ids.extend(["PAD" for _ in range(512 - token_len)])

        ret_dict["input_ids"] = torch.tensor([tokenizer.convert_tokens_to_ids(token_ids)], dtype=torch.long)
        ret_dict["token_type_ids"] = torch.tensor([[0 for _ in range(512)]], dtype=torch.long)
        ret_dict["attention_mask"] = torch.tensor([[1 if i < attention_size else 0 for i in range(512)]], dtype=torch.long)

    return ret_dict


### MAIN ###
if "__main__" == __name__:
    model_path = "../../model/low_precision/low_precision_score.pt"
    print(f"[model_checker] Checking Model - {model_path}")

    tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    model = torch.load(model_path)
    model.cuda()

    # Check output
    input_str = "아...카레를 먹었더니 왠지 요가를 할 수 있을 것 같은 기분."
    inputs = make_input_by_sentence(tokenizer, input_str=input_str)

    input_ids = inputs["input_ids"].to("cuda")
    token_type_ids = inputs["token_type_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")

    model.eval()
    outputs = model(input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask)

    preds = outputs.logits.detach().cpu().numpy()
    preds = np.argmax(preds, axis=2)
    preds = preds.reshape(512)

    label_list = TTA_NE_tags.keys()
    id2label = {i: label for i, label in enumerate(label_list)}

    pred_labels = [id2label[i] for i in preds]
    tokens = inputs["input_ids"].tolist()[0]
    print(tokenizer.convert_ids_to_tokens(tokens))
    print(tokens)
    print(pred_labels)