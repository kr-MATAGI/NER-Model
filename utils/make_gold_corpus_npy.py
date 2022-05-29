import pickle
from builtins import enumerate

import numpy as np

from ne_tag_def import ETRI_TAG
from pos_tag_def import NIKL_POS_TAG

from typing import List
from data_def import Sentence, NE, Morp

from transformers import AutoTokenizer

def conv_TTA_ne_category(sent_list: List[Sentence]):
    '''
    NE Type : 세분류 -> 대분류

    Args:
        sent_list: list of Sentence dataclass
    '''
    for sent_item in sent_list:
        for ne_item in sent_item.ne_list:
            conv_type = ""
            lhs_type = ne_item.type.split("_")[0]
            if "PS" in lhs_type:
                conv_type = "PS"
            elif "LC" in lhs_type:
                conv_type = "LC"
            elif "OG" in lhs_type:
                conv_type = "OG"
            elif "AF" in lhs_type:
                conv_type = "AF"
            elif "DT" in lhs_type:
                conv_type = "DT"
            elif "TI" in lhs_type:
                conv_type = "TI"
            elif "CV" in lhs_type:
                conv_type = "CV"
            elif "AM" in lhs_type:
                conv_type = "AM"
            elif "PT" in lhs_type:
                conv_type = "PT"
            elif "QT" in lhs_type:
                conv_type = "QT"
            elif "FD" in lhs_type:
                conv_type = "FD"
            elif "TR" in lhs_type:
                conv_type = "TR"
            elif "EV" in lhs_type:
                conv_type = "EV"
            elif "MT" in lhs_type:
                conv_type = "MT"
            elif "TM" in lhs_type:
                conv_type = "TM"

            if "" == conv_type:
                print(sent_item.text, "\n", lhs_type)
                return
            ne_item.type = conv_type
    return sent_list


def make_npy(tokenizer_name: str, src_list: List[Sentence], max_len: int=512):
    '''
    Note:
        Sentence + Named Entity + Morp Token Npy
    '''


    npy_dict = {
        "input_ids": [],
        "labels": [],
        "attention_mask": [],
        "token_type_ids": [],
        "seq_len": [],
        "pos_tag_ids": [],
    }
    pos_tag2ids = {v: k for k, v in NIKL_POS_TAG.items()}

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    for proc_idx, sent in enumerate(src_list):
        if 0 == (proc_idx % 1000):
            print(f"{proc_idx} Processing... {sent.text}")

        # if 321000 >= proc_idx: # [ 그, 근처, ##라드, ##라고] , [라, 드라, 고]
        #     continue

        text_tokens = tokenizer.tokenize(sent.text)
        labels = ["O"] * len(text_tokens)
        pos_tag_ids = [(["O"] * 2)] * len(text_tokens)


        # NE
        start_idx = 0
        for ne_item in sent.ne_list:
            ne_tokens = tokenizer.tokenize(ne_item.text)

            for s_idx in range(start_idx, len(text_tokens)):
                concat_token = text_tokens[s_idx:s_idx + len(ne_tokens)]

                if "".join(concat_token) == "".join(ne_tokens):
                    for l_idx in range(s_idx, s_idx + len(ne_tokens)):
                        if l_idx == s_idx:
                            labels[l_idx] = "B-" + ne_item.type
                        else:
                            labels[l_idx] = "I-" + ne_item.type
                    start_idx = s_idx
                    break
        ## end, ne_item loop

        # TEST
        # test_ne_print = [(x.text, x.type) for x in sent.ne_list]
        # id2la = {v: k for k, v in ETRI_TAG.items()}
        # print(test_ne_print)
        # for t, l in zip(text_tokens, labels):
        #     print(t, "\t", l)
        # input()

        # Morp
        is_all_ok = True
        pos_tag_idx = 0
        for word_item in sent.word_list:
            word_tokens = tokenizer.tokenize(word_item.form)
            matched_pos_tag_list = []
            for morp_item in sent.morp_list:
                if word_item.id == morp_item.word_id:
                    matched_pos_tag_list.append(morp_item)

            word_tokens = [x.replace("##", "") for x in word_tokens]
            concat_word_tok = []
            for word_tok in word_tokens:
                concat_word_tok.append(word_tok)

                for mps_idx, matched_pos_tag in enumerate(matched_pos_tag_list):
                    if "".join(concat_word_tok) == matched_pos_tag.form:
                        print(sent.text)
                        print(text_tokens)
                        print(concat_word_tok, matched_pos_tag.form)
                        input()
                        for _ in range(len(concat_word_tok)):
                            pos_tag_ids[pos_tag_idx] = matched_pos_tag.label
                            pos_tag_idx += 1
                        concat_word_tok = []
                        break
                    elif mps_idx < (len(matched_pos_tag_list) - 1):
                        if matched_pos_tag_list[mps_idx+1].label in ["ETM", "EF"]:
                            concat_pos_tag = matched_pos_tag_list[mps_idx].form + matched_pos_tag_list[mps_idx+1].form

                            if j2hcj(h2j("".join(concat_word_tok))) == j2hcj(h2j(concat_pos_tag)):
                                pos_tag_ids[pos_tag_idx] = matched_pos_tag_list[mps_idx].label
                                pos_tag_idx += 1
                                pos_tag_ids[pos_tag_idx] = matched_pos_tag_list[mps_idx+1].label
                                pos_tag_idx += 1

                                print(sent.text)
                                print(text_tokens)
                                print(concat_word_tok, matched_pos_tag.form)
                                input()

        # for wordpiece tokenization
        if not is_all_ok:
            continue


    #     for mp_item in sent.morp_list:
    #         mp_tokens = tokenizer.tokenize(mp_item.form)
    #         mp_tokens = [tok.replace("##", "") for tok in mp_tokens]
    #
    #         for s_idx in range(start_idx, len(text_tokens)):
    #             concat_token = text_tokens[s_idx:s_idx+len(mp_tokens)]
    #             concat_token = [tok.replace("##", "") for tok in concat_token]
    #
    #             if "".join(concat_token) == "".join(mp_tokens):
    #                 for l_idx in range(s_idx, s_idx + len(mp_tokens)):
    #                     try:
    #                         pos_tag_ids[l_idx] = mp_item.label
    #                     except:
    #                         print(text_tokens)
    #                         print(mp_tokens)
    #                         print(len(pos_tag_ids))
    #                         print(pos_tag_ids)
    #                         print(mp_item.label)
    #                 start_idx = s_idx
    #                 break
    #
    #     text_tokens.insert(0, "[CLS]")
    #     labels.insert(0, "O")
    #     pos_tag_ids.insert(0, "O")
    #
    #     valid_len = 0
    #     if max_len <= len(text_tokens):
    #         text_tokens = text_tokens[:max_len - 1]
    #         labels = labels[:max_len - 1]
    #         pos_tag_ids = pos_tag_ids[:max_len - 1]
    #
    #         text_tokens.append("[SEP]")
    #         labels.append("O")
    #         pos_tag_ids.append("O")
    #
    #         valid_len = max_len
    #     else:
    #         text_tokens.append("[SEP]")
    #         labels.append("O")
    #         pos_tag_ids.append("O")
    #
    #         valid_len = len(text_tokens)
    #         text_tokens = text_tokens + ["[PAD]"] * (max_len - valid_len)
    #         labels = labels + ["O"] * (max_len - valid_len)
    #         pos_tag_ids = pos_tag_ids + ["O"] * (max_len - valid_len)
    #
    #     attention_mask = ([1] * valid_len) + ([0] * (max_len - valid_len))
    #     token_type_ids = [0] * max_len
    #     input_ids = tokenizer.convert_tokens_to_ids(text_tokens)
    #     labels = [ETRI_TAG[x] for x in labels]
    #     pos_tag_ids = [pos_tag2ids[x] for x in pos_tag_ids]
    #
    #     assert len(input_ids) == max_len, f"{input_ids} + {len(input_ids)}"
    #     assert len(labels) == max_len, f"{labels} + {len(labels)}"
    #     assert len(attention_mask) == max_len, f"{attention_mask} + {len(attention_mask)}"
    #     assert len(token_type_ids) == max_len, f"{token_type_ids} + {len(token_type_ids)}"
    #     assert len(pos_tag_ids) == max_len, f"{pos_tag_ids} + {len(pos_tag_ids)}"
    #
    #     npy_dict["input_ids"].append(input_ids)
    #     npy_dict["labels"].append(labels)
    #     npy_dict["attention_mask"].append(attention_mask)
    #     npy_dict["token_type_ids"].append(token_type_ids)
    #
    #     # for pack_padded_sequence
    #     npy_dict["seq_len"].append(valid_len)
    #
    #     # add pos tag info
    #     npy_dict["pos_tag_ids"].append(pos_tag_ids)
    #
    # # convert list to numpy
    # npy_dict["input_ids"] = np.array(npy_dict["input_ids"])
    # npy_dict["labels"] = np.array(npy_dict["labels"])
    # npy_dict["attention_mask"] = np.array(npy_dict["attention_mask"])
    # npy_dict["token_type_ids"] = np.array(npy_dict["token_type_ids"])
    # npy_dict["seq_len"] = np.array(npy_dict["seq_len"])
    # npy_dict["pos_tag_ids"] = np.array(npy_dict["pos_tag_ids"])
    # print(f"input_ids.shape: {npy_dict['input_ids'].shape}")
    # print(f"labels.shape: {npy_dict['labels'].shape}")
    # print(f"attention_mask.shape: {npy_dict['attention_mask'].shape}")
    # print(f"token_type_ids.shape: {npy_dict['token_type_ids'].shape}")
    # print(f"seq_len.shape: {npy_dict['seq_len'].shape}")
    # print(f"pos_tag_ids.shape: {npy_dict['pos_tag_ids'].shape}")
    #
    # split_size = int(len(src_list) * 0.1)
    # train_size = split_size * 7
    # valid_size = train_size + split_size
    #
    # train_np = [npy_dict["input_ids"][:train_size], npy_dict["labels"][:train_size],
    #             npy_dict["attention_mask"][:train_size], npy_dict["token_type_ids"][:train_size]]
    # train_np = np.stack(train_np, axis=-1)
    # train_seq_len_np = npy_dict["seq_len"][:train_size]
    # train_pos_tag_np = npy_dict["pos_tag_ids"][:train_size]
    # print(f"train_np.shape: {train_np.shape}")
    # print(f"train_seq_len_np.shape: {train_seq_len_np.shape}")
    # print(f"train_pos_tag_ids_np.shape: {train_pos_tag_np.shape}")
    #
    # dev_np = [npy_dict["input_ids"][train_size:valid_size], npy_dict["labels"][train_size:valid_size],
    #           npy_dict["attention_mask"][train_size:valid_size], npy_dict["token_type_ids"][train_size:valid_size]]
    # dev_np = np.stack(dev_np, axis=-1)
    # dev_seq_len_np = npy_dict["seq_len"][train_size:valid_size]
    # dev_pos_tag_np = npy_dict["pos_tag_ids"][train_size:valid_size]
    # print(f"dev_np.shape: {dev_np.shape}")
    # print(f"dev_seq_len_np.shape: {dev_seq_len_np.shape}")
    # print(f"dev_pos_tag_ids_np.shape: {dev_pos_tag_np.shape}")
    #
    # test_np = [npy_dict["input_ids"][valid_size:], npy_dict["labels"][valid_size:],
    #            npy_dict["attention_mask"][valid_size:], npy_dict["token_type_ids"][valid_size:]]
    # test_np = np.stack(test_np, axis=-1)
    # test_seq_len_np = npy_dict["seq_len"][valid_size:]
    # test_pos_tag_np = npy_dict["pos_tag_ids"][valid_size:]
    # print(f"test_np.shape: {test_np.shape}")
    # print(f"test_seq_len_np.shape: {test_seq_len_np.shape}")
    # print(f"test_pos_tag_ids_np.shape: {test_pos_tag_np.shape}")
    #
    # # save
    # np.save("../data/npy/old_nikl/128/train", train_np)
    # np.save("../data/npy/old_nikl/128/dev", dev_np)
    # np.save("../data/npy/old_nikl/128/test", test_np)
    #
    # # save seq_len
    # np.save("../data/npy/old_nikl/128/train_seq_len", train_seq_len_np)
    # np.save("../data/npy/old_nikl/128/dev_seq_len", dev_seq_len_np)
    # np.save("../data/npy/old_nikl/128/test_seq_len", test_seq_len_np)
    #
    # # save pos_tag
    # np.save("../data/npy/old_nikl/128/train_pos_tag", train_pos_tag_np)
    # np.save("../data/npy/old_nikl/128/dev_pos_tag", dev_pos_tag_np)
    # np.save("../data/npy/old_nikl/128/test_pos_tag", test_pos_tag_np)

### MAIN ###
if "__main__" == __name__:
    print("NPY MAKER !")

    # TEST
    tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-small-v3-discriminator")
    a = "고인의 삼남 최만립(81)씨는 법정에서 \"64년간 통한의 세월을 보냈는데 아버지 명예를 회복해 기쁘기 한량없다\"고 말했다."
    print(tokenizer.tokenize(a))


    exit()

    all_sent_list = []
    with open("../data/pkl/ne_mp_old_nikl.pkl", mode="rb") as pkl_file:
        all_sent_list = pickle.load(pkl_file)
        print(f"[npy_maker][__main__] all_sent_list size: {len(all_sent_list)}")
    all_sent_list = conv_TTA_ne_category(all_sent_list)
    print(f"main__: all_sent_list.len: {len(all_sent_list)}") # 371,571

    # make npy
    make_npy(tokenizer_name="monologg/koelectra-small-v3-discriminator", src_list=all_sent_list, max_len=128)