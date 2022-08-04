import copy
import random
import numpy as np
import pickle

from ne_tag_def import ETRI_TAG
from pos_tag_def import NIKL_POS_TAG

from typing import List, Dict, Tuple
from data_def import Sentence, NE, Morp

from transformers import AutoTokenizer, ElectraTokenizer
from dict_utils import Dict_Item, Word_Info, Sense_Info, make_dict_hash_table

### global
random.seed(42)
np.random.seed(42)

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
            else:
                print(f"What is {lhs_type}")
                return

            if "" == conv_type:
                print(sent_item.text, "\n", lhs_type)
                return
            ne_item.type = conv_type
    return sent_list

def save_npy_dict(npy_dict: Dict[str, List], src_list_len):
    # convert list to numpy
    npy_dict["input_ids"] = np.array(npy_dict["input_ids"])
    npy_dict["labels"] = np.array(npy_dict["labels"])
    npy_dict["attention_mask"] = np.array(npy_dict["attention_mask"])
    npy_dict["token_type_ids"] = np.array(npy_dict["token_type_ids"])
    npy_dict["seq_len"] = np.array(npy_dict["seq_len"])
    npy_dict["pos_tag_ids"] = np.array(npy_dict["pos_tag_ids"])
    npy_dict["span_ids"] = np.array(npy_dict["span_ids"])
    npy_dict["eojeol_ids"] = np.array(npy_dict["eojeol_ids"])
    print(f"input_ids.shape: {npy_dict['input_ids'].shape}")
    print(f"labels.shape: {npy_dict['labels'].shape}")
    print(f"attention_mask.shape: {npy_dict['attention_mask'].shape}")
    print(f"token_type_ids.shape: {npy_dict['token_type_ids'].shape}")
    print(f"seq_len.shape: {npy_dict['seq_len'].shape}")
    print(f"pos_tag_ids.shape: {npy_dict['pos_tag_ids'].shape}")
    print(f"span_ids.shape: {npy_dict['span_ids'].shape}")
    print(f"eojeol_ids.shape: {npy_dict['eojeol_ids'].shape}")

    split_size = int(src_list_len * 0.1)
    train_size = split_size * 7
    valid_size = train_size + split_size

    train_np = [npy_dict["input_ids"][:train_size], npy_dict["labels"][:train_size],
                npy_dict["attention_mask"][:train_size], npy_dict["token_type_ids"][:train_size]]
    train_np = np.stack(train_np, axis=-1)
    train_seq_len_np = npy_dict["seq_len"][:train_size]
    train_pos_tag_np = npy_dict["pos_tag_ids"][:train_size]
    train_span_ids_np = npy_dict["span_ids"][:train_size]
    train_eojeol_ids_np = npy_dict["eojeol_ids"][:train_size]
    print(f"train_np.shape: {train_np.shape}")
    print(f"train_seq_len_np.shape: {train_seq_len_np.shape}")
    print(f"train_pos_tag_ids_np.shape: {train_pos_tag_np.shape}")
    print(f"train_span_ids_np.shape: {train_span_ids_np.shape}")
    print(f"train_eojeol_ids_np.shape: {train_eojeol_ids_np.shape}")

    dev_np = [npy_dict["input_ids"][train_size:valid_size], npy_dict["labels"][train_size:valid_size],
              npy_dict["attention_mask"][train_size:valid_size], npy_dict["token_type_ids"][train_size:valid_size]]
    dev_np = np.stack(dev_np, axis=-1)
    dev_seq_len_np = npy_dict["seq_len"][train_size:valid_size]
    dev_pos_tag_np = npy_dict["pos_tag_ids"][train_size:valid_size]
    dev_span_ids_np = npy_dict["span_ids"][train_size:valid_size]
    dev_eojeol_ids_np = npy_dict["eojeol_ids"][train_size:valid_size]
    print(f"dev_np.shape: {dev_np.shape}")
    print(f"dev_seq_len_np.shape: {dev_seq_len_np.shape}")
    print(f"dev_pos_tag_ids_np.shape: {dev_pos_tag_np.shape}")
    print(f"dev_span_ids_np.shape: {dev_span_ids_np.shape}")
    print(f"dev_eojeol_ids_np.shape: {dev_eojeol_ids_np.shape}")

    test_np = [npy_dict["input_ids"][valid_size:], npy_dict["labels"][valid_size:],
               npy_dict["attention_mask"][valid_size:], npy_dict["token_type_ids"][valid_size:]]
    test_np = np.stack(test_np, axis=-1)
    test_seq_len_np = npy_dict["seq_len"][valid_size:]
    test_pos_tag_np = npy_dict["pos_tag_ids"][valid_size:]
    test_span_ids_np = npy_dict["span_ids"][valid_size:]
    test_eojeol_ids_np = npy_dict["eojeol_ids"][valid_size:]
    print(f"test_np.shape: {test_np.shape}")
    print(f"test_seq_len_np.shape: {test_seq_len_np.shape}")
    print(f"test_pos_tag_ids_np.shape: {test_pos_tag_np.shape}")
    print(f"test_span_ids_np.shape: {test_span_ids_np.shape}")
    print(f"test_eojeol_ids_np.shape: {test_eojeol_ids_np.shape}")

    # save
    np.save("../data/npy/old_nikl/electra/train", train_np)
    np.save("../data/npy/old_nikl/electra/dev", dev_np)
    np.save("../data/npy/old_nikl/electra/test", test_np)

    # save seq_len
    np.save("../data/npy/old_nikl/electra/train_seq_len", train_seq_len_np)
    np.save("../data/npy/old_nikl/electra/dev_seq_len", dev_seq_len_np)
    np.save("../data/npy/old_nikl/electra/test_seq_len", test_seq_len_np)

    # save pos_tag
    np.save("../data/npy/old_nikl/electra/train_pos_tag", train_pos_tag_np)
    np.save("../data/npy/old_nikl/electra/dev_pos_tag", dev_pos_tag_np)
    np.save("../data/npy/old_nikl/electra/test_pos_tag", test_pos_tag_np)

    # save span_ids
    np.save("../data/npy/old_nikl/electra/train_span_ids", train_span_ids_np)
    np.save("../data/npy/old_nikl/electra/dev_span_ids", dev_span_ids_np)
    np.save("../data/npy/old_nikl/electra/test_span_ids", test_span_ids_np)

    # save eojeol_ids
    np.save("../data/npy/old_nikl/electra/train_eojeol_ids", train_eojeol_ids_np)
    np.save("../data/npy/old_nikl/electra/dev_eojeol_ids", dev_eojeol_ids_np)
    np.save("../data/npy/old_nikl/electra/test_eojeol_ids", test_eojeol_ids_np)

def make_wordpiece_npy(tokenizer_name:str, src_list: List[Sentence], max_len: int=128):
    random.shuffle(src_list)

    npy_dict = {
        "input_ids": [],
        "labels": [],
        "attention_mask": [],
        "token_type_ids": [],
        "seq_len": [],
        "pos_tag_ids": [],
        "span_ids": [],
        "eojeol_ids": []
    }
    pos_tag2ids = {v: int(k) for k, v in NIKL_POS_TAG.items()}

    tokenizer = ElectraTokenizer.from_pretrained(tokenizer_name)
    for proc_idx, sent in enumerate(src_list):
        if 0 == (proc_idx % 1000):
            print(f"{proc_idx} Processing... {sent.text}")

        # if sent.text != "소비자원에서 어떤 일을 해 주면 정말 소비자들한테 필요한 사업이 될까에 대해서 이런 아이디어를 공모를 해서":
        #     continue
        
        text_tokens = tokenizer.tokenize(sent.text)

        labels = ["O"] * len(text_tokens)
        pos_tag_ids = []
        span_ids = []
        eojeol_ids = []
        for _ in range(len(text_tokens)):
            pos_tag_ids.append(["O"] * 3)
            span_ids.append(0)

        # NE
        start_idx = 0
        # Testing
        # text_tokens = tokenizer.tokenize("어 가필드 가필드 고양이 그래서 그게 두개 섞인 거 같은데 이제 나는 걔를 얘기")
        # a = NE(id=1, text="가필드", type="AF")
        # b = NE(id=2, text="가필드", type="AF")
        # c = NE(id=3, text="고양이", type="AM")
        # sent.ne_list = [a, b, c]
        for ne_item in sent.ne_list:
            is_find = False
            for s_idx in range(start_idx, len(text_tokens)):
                if is_find:
                    break
                for word_cnt in range(0, len(text_tokens) - s_idx + 1):
                    concat_text_tokens = text_tokens[s_idx:s_idx + word_cnt]
                    concat_text_tokens = [x.replace("##", "") for x in concat_text_tokens]

                    if "".join(concat_text_tokens) == ne_item.text.replace(" ", ""):
                        # print("A : ", concat_text_tokens, ne_item.text, s_idx)

                        # BIO Tagging
                        for bio_idx in range(s_idx, s_idx + word_cnt):
                            if bio_idx == s_idx:
                                labels[bio_idx] = "B-" + ne_item.type
                            else:
                                labels[bio_idx] = "I-" + ne_item.type

                        is_find = True
                        start_idx = s_idx + word_cnt
                        break
        ## end, ne_item loop

        # TEST and Print
        # test_ne_print = [(x.text, x.type) for x in sent.ne_list]
        # print(test_ne_print)
        # for t, l in zip(text_tokens, labels):
        #    print(t, "\t", l)
        # input()

        # Morp
        start_idx = 0
        # Test
        test_str = "▷잘못된 체제에다 무능하고 부패한 지도자를 만나 허덕이는 북한 주민도 해외에서 외화벌이에 나서고 있다."
        # if test_str != sent.text:
        #     continue
        for morp_item in sent.morp_list:
            is_find = False
            split_morp_label_item = morp_item.label.split("+")

            for s_idx in range(start_idx, len(text_tokens)):
                if is_find:
                    break
                for word_size in range(0, len(text_tokens) - s_idx + 1):
                    concat_text_tokens = text_tokens[s_idx:s_idx + word_size]
                    concat_text_tokens = [x.replace("##", "") for x in concat_text_tokens]
                    concat_str = "".join(concat_text_tokens)
                    if len(concat_str) > len(morp_item.form):
                        break
                    if pos_tag_ids[s_idx][0] != "O":
                        continue
                    # print(concat_str, morp_item.form, morp_item.label)
                    if concat_str == morp_item.form:
                        #print("A:", concat_str, morp_item.form, morp_item.label)
                        for p_idx in range(s_idx, s_idx+word_size):
                            for sp_idx, sp_item in enumerate(split_morp_label_item):
                                if 2 < sp_idx:
                                    break
                                pos_tag_ids[p_idx][sp_idx] = sp_item
                                if "" == sp_item: # Exception
                                    pos_tag_ids[p_idx][sp_idx] = "O"
                        is_find = True
                        #start_idx = s_idx + word_size
                        break
        # TEST and Print
        # test_ne_print = [(x.form, x.label) for x in sent.morp_list]
        # id2pos = {v: k for k, v in NIKL_POS_TAG.items()}
        # print(test_ne_print)
        # for t, l in zip(text_tokens, pos_tag_ids):
        #    print(t, "\t", l)
        # input()

        # span
        '''
        # print(sent.text)
        # print(text_tokens)
        nn_pos_list = ["NNG", "NNP"] # "NNB", "NP"
        pre_pos_list = ["MMA", "VA", "ETM"]
        post_pos_list = ["XSN"]
        find_word_list = [] # 사전에서 찾은 것들
        candi_list = []
        for morp_item in sent.morp_list:
            # print(morp_item.form, morp_item.label)

            if (morp_item.label in nn_pos_list) or (morp_item.label in post_pos_list):
                candi_list.append(morp_item.form)
                concat_cadi = "".join(candi_list)
                if concat_cadi[0] in ex_dictionary.keys():
                    dict_item_list = ex_dictionary[concat_cadi[0]]
                    dict_item_list = [x.word_info.word for x in dict_item_list if x.sense_info.pos == "명사"]
                    if concat_cadi in dict_item_list:
                        find_word_list.append(concat_cadi)
            elif morp_item.label in pre_pos_list:
                # 현재 label은 보통 명사 앞에 붙기 때문에 현재 넣은거 검색하고 새로 넣음
                concat_cadi = "".join(candi_list)
                if (0 < len(concat_cadi)) and (concat_cadi[0] in ex_dictionary.keys()):
                    dict_item_list = ex_dictionary[concat_cadi[0]]
                    dict_item_list = [x.word_info.word for x in dict_item_list if x.sense_info.pos == "명사"]
                    if (concat_cadi in dict_item_list) and (concat_cadi not in find_word_list):
                        find_word_list.append(concat_cadi)
                candi_list.clear()
                candi_list.append(morp_item.form)
            else:
                candi_list.clear()
        # print(find_word_list)

        # 찾은 사전 단어들에 대한 범위 지정
        span_num = 1 # special token is '0'
        for t_idx in range(0, len(text_tokens)):
            is_find = False
            for find_word in find_word_list:
                find_word_tokens = tokenizer.tokenize(find_word)
                find_word_tokens = [x.replace("##", "") for x in find_word_tokens]
                if len(text_tokens) < t_idx+len(find_word_tokens):
                    continue
                curr_token = text_tokens[t_idx:t_idx + len(find_word_tokens)]
                curr_token = [x.replace("##", "") for x in curr_token]
                if curr_token == find_word_tokens:
                    # print("1 ", curr_token)
                    # print("2 ", find_word_tokens)
                    is_find = True
                    for s_idx in range(t_idx, t_idx + len(curr_token)):
                        # print(text_tokens[s_idx], span_num)
                        span_ids[s_idx] = span_num
            if not is_find:
                if 0 == span_ids[t_idx]:
                    span_ids[t_idx] = span_num
                    span_num += 1
            else:
                span_num += 1
        '''
        # print(span_ids)
        # print(len(text_tokens))
        # print(len(span_ids))
        # input()

        text_tokens.insert(0, "[CLS]")
        labels.insert(0, "O")
        pos_tag_ids.insert(0, (["O"] * 3))
        span_ids.insert(0, 0)

        valid_len = 0
        if max_len <= len(text_tokens):
            text_tokens = text_tokens[:max_len - 1]
            labels = labels[:max_len - 1]
            pos_tag_ids = pos_tag_ids[:max_len - 1]
            span_ids = span_ids[:max_len - 1]

            text_tokens.append("[SEP]")
            labels.append("O")
            pos_tag_ids.append((["O"] * 3))
            span_ids.append(0)

            valid_len = max_len
        else:
            text_tokens.append("[SEP]")
            labels.append("O")
            pos_tag_ids.append((["O"] * 3))
            span_ids.append(0)

            valid_len = len(text_tokens)
            text_tokens = text_tokens + ["[PAD]"] * (max_len - valid_len)
            labels = labels + ["O"] * (max_len - valid_len)
            for _ in range(max_len - valid_len):
                pos_tag_ids.append((["O"] * 3))
                span_ids.append(0)

        attention_mask = ([1] * valid_len) + ([0] * (max_len - valid_len))
        token_type_ids = [0] * max_len
        input_ids = tokenizer.convert_tokens_to_ids(text_tokens)
        labels = [ETRI_TAG[x] for x in labels]
        for i in range(max_len):
            pos_tag_ids[i] = [pos_tag2ids[x] for x in pos_tag_ids[i]]

        # eojeol boundary (22.08.01)
        eojeol_num = 1
        eojeol_ids = [0 for _ in range(len(text_tokens))]
        special_tokens = ["[CLS]", "[SEP]", "[PAD]", "[UNK]"]
        for tdx, txt_tok in enumerate(text_tokens):
            if txt_tok in special_tokens:
                eojeol_ids[tdx] = eojeol_num
                eojeol_num += 1
            elif "##" in txt_tok:
                eojeol_ids[tdx] = eojeol_num
            else:
                eojeol_num += 1
                eojeol_ids[tdx] = eojeol_num
        # for a, b in zip(text_tokens, eojeol_ids):
        #     print(a, b)
        # input()

        assert len(input_ids) == max_len, f"{input_ids} + {len(input_ids)}"
        assert len(labels) == max_len, f"{labels} + {len(labels)}"
        assert len(attention_mask) == max_len, f"{attention_mask} + {len(attention_mask)}"
        assert len(token_type_ids) == max_len, f"{token_type_ids} + {len(token_type_ids)}"
        assert len(pos_tag_ids) == max_len, f"{pos_tag_ids} + {len(pos_tag_ids)}"
        assert len(eojeol_ids) == max_len, f"{eojeol_ids} + {len(eojeol_ids)}"

        npy_dict["input_ids"].append(input_ids)
        npy_dict["labels"].append(labels)
        npy_dict["attention_mask"].append(attention_mask)
        npy_dict["token_type_ids"].append(token_type_ids)

        # for pack_padded_sequence
        npy_dict["seq_len"].append(valid_len)

        # add pos tag info
        npy_dict["pos_tag_ids"].append(pos_tag_ids)

        # add span ids
        npy_dict["span_ids"].append(span_ids)

        # add eojeol boundary
        npy_dict["eojeol_ids"].append(eojeol_ids)

        # save
    save_npy_dict(npy_dict, src_list_len=len(src_list))


def make_pos_tag_npy(tokenizer_name: str, src_list: List[Sentence], max_len: int=512):
    '''
    Note:
        Sentence + Named Entity + Morp Token Npy

        tokenization base is pos tagging
    '''

    npy_dict = {
        "input_ids": [],
        "labels": [],
        "attention_mask": [],
        "token_type_ids": [],
        "seq_len": [],
        "pos_tag_ids": [],
    }
    pos_tag2ids = {v: int(k) for k, v in NIKL_POS_TAG.items()}

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    for proc_idx, sent in enumerate(src_list):
        if 0 == (proc_idx % 1000):
            print(f"{proc_idx} Processing... {sent.text}")

        text_tokens = []
        for mp_item in sent.morp_list:
            mp_tokens = tokenizer.tokenize(mp_item.form)
            text_tokens.extend(mp_tokens)
        labels = ["O"] * len(text_tokens)
        pos_tag_ids = []
        for _ in range(len(text_tokens)):
            pos_tag_ids.append(["O"] * 3)

        # NE
        start_idx = 0
        for ne_item in sent.ne_list:
            is_find = False
            for s_idx in range(start_idx, len(text_tokens)):
                if is_find:
                    break
                for word_cnt in range(0, len(text_tokens) - s_idx + 1):
                    concat_text_tokens = text_tokens[s_idx:s_idx+word_cnt]
                    concat_text_tokens = [x.replace("##", "") for x in concat_text_tokens]

                    if "".join(concat_text_tokens) == ne_item.text.replace(" ", ""):
                        # print("A : ", concat_text_tokens, ne_item.text)

                        # BIO Tagging
                        for bio_idx in range(s_idx, s_idx+word_cnt):
                            if bio_idx == s_idx:
                                labels[bio_idx] = "B-" + ne_item.type
                            else:
                                labels[bio_idx] = "I-" + ne_item.type

                        is_find = True
                        start_idx = s_idx
                        break
        ## end, ne_item loop

        # TEST and Print
        # test_ne_print = [(x.text, x.type) for x in sent.ne_list]
        # id2la = {v: k for k, v in ETRI_TAG.items()}
        # print(test_ne_print)
        # for t, l in zip(text_tokens, labels):
        #     print(t, "\t", l)

        # Morp
        pos_idx = 0
        for morp_item in sent.morp_list:
            mp_tokens = tokenizer.tokenize(morp_item.form)

            split_morp_label_item = morp_item.label.split("+")
            for _ in range(len(mp_tokens)):
                for sp_idx, split_label in enumerate(split_morp_label_item):
                    if 2 < sp_idx:
                        continue
                    pos_tag_ids[pos_idx][sp_idx] = split_label
                    if "" == split_label:
                        pos_tag_ids[pos_idx][sp_idx] = "O"
                pos_idx += 1

        text_tokens.insert(0, "[CLS]")
        labels.insert(0, "O")
        pos_tag_ids.insert(0, (["O"] * 3))

        valid_len = 0
        if max_len <= len(text_tokens):
            text_tokens = text_tokens[:max_len - 1]
            labels = labels[:max_len - 1]
            pos_tag_ids = pos_tag_ids[:max_len - 1]

            text_tokens.append("[SEP]")
            labels.append("O")
            pos_tag_ids.append((["O"] * 3))

            valid_len = max_len
        else:
            text_tokens.append("[SEP]")
            labels.append("O")
            pos_tag_ids.append((["O"] * 3))

            valid_len = len(text_tokens)
            text_tokens = text_tokens + ["[PAD]"] * (max_len - valid_len)
            labels = labels + ["O"] * (max_len - valid_len)
            for _ in range(max_len - valid_len):
                pos_tag_ids.append((["O"] * 3))

        attention_mask = ([1] * valid_len) + ([0] * (max_len - valid_len))
        token_type_ids = [0] * max_len
        input_ids = tokenizer.convert_tokens_to_ids(text_tokens)
        labels = [ETRI_TAG[x] for x in labels]
        for i in range(max_len):
            pos_tag_ids[i] = [pos_tag2ids[x] for x in pos_tag_ids[i]]

        assert len(input_ids) == max_len, f"{input_ids} + {len(input_ids)}"
        assert len(labels) == max_len, f"{labels} + {len(labels)}"
        assert len(attention_mask) == max_len, f"{attention_mask} + {len(attention_mask)}"
        assert len(token_type_ids) == max_len, f"{token_type_ids} + {len(token_type_ids)}"
        assert len(pos_tag_ids) == max_len, f"{pos_tag_ids} + {len(pos_tag_ids)}"

        npy_dict["input_ids"].append(input_ids)
        npy_dict["labels"].append(labels)
        npy_dict["attention_mask"].append(attention_mask)
        npy_dict["token_type_ids"].append(token_type_ids)

        # for pack_padded_sequence
        npy_dict["seq_len"].append(valid_len)

        # add pos tag info
        npy_dict["pos_tag_ids"].append(pos_tag_ids)

    # save
    save_npy_dict(npy_dict, src_list_len=len(src_list))
####

def make_eojeol_datasets_npy(tokenizer_name: str, src_list: List[Sentence],
                             max_len: int=128, eojeol_max_len: int=50, debug_mode: bool=False):
    random.shuffle(src_list)

    npy_dict = {
        "input_ids": [],
        "labels": [],
        "attention_mask": [],
        "token_type_ids": [],
        "seq_len": [],
        "pos_tag_ids": [],
        "eojeol_ids": []
    }
    pos_tag2ids = {v: int(k) for k, v in NIKL_POS_TAG.items()}

    tokenizer = ElectraTokenizer.from_pretrained(tokenizer_name)
    for proc_idx, src_item in enumerate(src_list):
        if 0 == (proc_idx % 1000):
            print(f"{proc_idx} Processing... {src_item.text}")

        # make (word, token, pos) pair
        text_tokens = []
        word_tokens_pos_pair_list: List[Tuple[str, List[str], Tuple[int, int]]] = [] # [(word, [tokens], (begin, end))]
        for word_idx, word_item in enumerate(src_item.word_list):
            form_tokens = tokenizer.tokenize(word_item.form)
            text_tokens.extend(form_tokens)
            word_tokens_pos_pair_list.append((word_item.form, form_tokens, (word_item.begin, word_item.end)))

        # make NE labels
        labels_ids = [ETRI_TAG["O"]] * len(word_tokens_pos_pair_list)
        b_check_use_eojeol = [False for _ in range(len(word_tokens_pos_pair_list))]
        is_B_tag = True  # BIO 중 B-[tag]
        for wtp_idx, word_token_pos in enumerate(word_tokens_pos_pair_list):
            if b_check_use_eojeol[wtp_idx]:  # 이미 사용한 어절
                continue
            for ne_idx, ne_item in enumerate(src_item.ne_list):
                if word_token_pos[-1][1] < ne_item.end: # 현재 어절보다 뒤에 있는 개체명
                    break
                if ne_item.begin >= word_token_pos[-1][0] and ne_item.end <= word_token_pos[-1][1]:
                    b_check_use_eojeol[wtp_idx] = True
                    # 개체명 범위에 포함되는 경우
                    if is_B_tag:
                        labels_ids[wtp_idx] = ETRI_TAG["B-"+ne_item.type]
                        is_B_tag = False
                    else:
                        labels_ids[wtp_idx] = ETRI_TAG["I-"+ne_item.type]
                    break
                else:
                    is_B_tag = True

        # POS
        pos_tag_ids = [] # [[POS, * 4]]
        cur_pos_tag = [pos_tag2ids["O"]] * 10
        add_idx = 0
        cur_word_id = 1
        for ps_idx, morp_item in enumerate(src_item.morp_list):
            if cur_word_id != morp_item.word_id:
                cur_word_id = morp_item.word_id
                pos_tag_ids.append(copy.deepcopy(cur_pos_tag))
                cur_pos_tag = [pos_tag2ids["O"]] * 10
                add_idx = 0
            if 10 > add_idx:
                cur_pos_tag[add_idx] = pos_tag2ids[morp_item.label]
                add_idx += 1
        # last word_id
        pos_tag_ids.append(copy.deepcopy(cur_pos_tag))

        # eojeol boundary
        eojeol_boundary_list: List[int] = []
        for wtp_ids, wtp_item in enumerate(word_tokens_pos_pair_list):
            token_size = len(wtp_item[1])
            eojeol_boundary_list.append(token_size)

        # Sequence Length
        # 토큰 단위
        valid_token_len = 0
        text_tokens.insert(0, "[CLS]")
        text_tokens.append("[SEP]")
        if max_len <= len(text_tokens):
            text_tokens = text_tokens[:max_len-1]
            text_tokens.append("[SEP]")
            valid_token_len = max_len
        else:
            valid_token_len = len(text_tokens)
            text_tokens += ["[PAD]"] * (max_len - valid_token_len)

        attention_mask = ([1] * valid_token_len) + ([0] * (max_len - valid_token_len))
        token_type_ids = [0] * max_len
        input_ids = tokenizer.convert_tokens_to_ids(text_tokens)

        # 어절 단위
        # label_ids
        valid_eojeol_len = 0
        labels_ids.insert(0, ETRI_TAG["O"])
        if eojeol_max_len <= len(labels_ids):
            labels_ids = labels_ids[:eojeol_max_len-1]
            labels_ids.append(ETRI_TAG["O"])
            valid_eojeol_len = eojeol_max_len
        else:
            labels_ids_size = len(labels_ids)
            valid_eojeol_len = labels_ids_size + 1 # + [SEP]
            for _ in range(eojeol_max_len - labels_ids_size):
                labels_ids.append(ETRI_TAG["O"])

        # pos_tag_ids
        pos_tag_ids.insert(0, [pos_tag2ids["O"]] * 10) # [CLS]
        if eojeol_max_len <= len(pos_tag_ids):
            pos_tag_ids = pos_tag_ids[:eojeol_max_len-1]
            pos_tag_ids.append([pos_tag2ids["O"]] * 10) # [SEP]
        else:
            # pos_tag_ids.append([pos_tag2ids["O"]] * 10)  # [SEP]
            pos_tag_ids_size = len(pos_tag_ids)
            for _ in range(eojeol_max_len - pos_tag_ids_size):
                pos_tag_ids.append([pos_tag2ids["O"]] * 10)

        # eojeol_ids
        eojeol_boundary_list.insert(0, 1) # [CLS]
        eojeol_boundary_list.append(1)  # [SEP]
        if eojeol_max_len <= len(eojeol_boundary_list):
            eojeol_boundary_list = eojeol_boundary_list[:eojeol_max_len-1]
            eojeol_boundary_list.append(1) # [SEP]
        else:
            eojeol_boundary_size = len(eojeol_boundary_list)
            eojeol_boundary_list += [0] * (eojeol_max_len - eojeol_boundary_size)

        assert len(input_ids) == max_len, f"{len(input_ids)} + {input_ids}"
        assert len(attention_mask) == max_len, f"{len(attention_mask)} + {attention_mask}"
        assert len(token_type_ids) == max_len, f"{len(token_type_ids)} + {token_type_ids}"
        assert len(labels_ids) == eojeol_max_len, f"{len(labels_ids)} + {labels_ids}"
        assert len(pos_tag_ids) == eojeol_max_len, f"{len(pos_tag_ids)} + {pos_tag_ids}"
        assert len(eojeol_boundary_list) == eojeol_max_len, f"{len(eojeol_boundary_list)} + {eojeol_boundary_list}"

        # add to npy_dict
        npy_dict["input_ids"].append(input_ids)
        npy_dict["attention_mask"].append(attention_mask)
        npy_dict["token_type_ids"].append(token_type_ids)
        npy_dict["seq_len"].append(valid_eojeol_len) # eojeol !
        npy_dict["labels"].append(labels_ids)
        npy_dict["pos_tag_ids"].append(pos_tag_ids)
        npy_dict["eojeol_ids"].append(eojeol_boundary_list)

        # debug_mode
        if debug_mode:
            print("Unit: WordPiece Token")
            print(f"text_tokens: {text_tokens}")
            for ii, am, tti in zip(input_ids, attention_mask, token_type_ids):
                print(ii, am, tti)
            print("Unit: Eojeol")
            print(f"seq_len: {valid_eojeol_len} : {len(word_tokens_pos_pair_list)}")
            print(f"label_ids.len: {len(labels_ids)}, pos_tag_ids.len: {len(pos_tag_ids)}")
            print(f"{word_tokens_pos_pair_list}")
            print(f"NE: {src_item.ne_list}")
            print(f"eojeol boundary: {eojeol_boundary_list}")
            for la, pti in zip(labels_ids, pos_tag_ids):
                print(la, pti)
            input()

    # save npy_dict
    save_eojeol_npy_dict(npy_dict, len(src_list))

def save_eojeol_npy_dict(npy_dict: Dict[str, List], src_list_len):
    npy_dict["input_ids"] = np.array(npy_dict["input_ids"])
    npy_dict["attention_mask"] = np.array(npy_dict["attention_mask"])
    npy_dict["token_type_ids"] = np.array(npy_dict["token_type_ids"])
    npy_dict["labels"] = np.array(npy_dict["labels"])
    npy_dict["seq_len"] = np.array(npy_dict["seq_len"])
    npy_dict["pos_tag_ids"] = np.array(npy_dict["pos_tag_ids"])
    npy_dict["eojeol_ids"] = np.array(npy_dict["eojeol_ids"])
    # wordpiece 토큰 단위
    print(f"Unit: Tokens")
    print(f"attention_mask.shape: {npy_dict['attention_mask'].shape}")
    print(f"token_type_ids.shape: {npy_dict['token_type_ids'].shape}")
    print(f"seq_len.shape: {npy_dict['seq_len'].shape}")

    # 어절 단위
    print(f"Unit: Eojoel")
    print(f"input_ids.shape: {npy_dict['input_ids'].shape}")
    print(f"labels.shape: {npy_dict['labels'].shape}")
    print(f"pos_tag_ids.shape: {npy_dict['pos_tag_ids'].shape}")
    print(f"eojeol_ids.shape: {npy_dict['eojeol_ids'].shape}")

    # train/dev/test 분할
    split_size = int(src_list_len * 0.1)
    train_size = split_size * 7
    valid_size = train_size + split_size

    # train
    train_np = [npy_dict["input_ids"][:train_size],
                npy_dict["attention_mask"][:train_size],
                npy_dict["token_type_ids"][:train_size]]
    train_np = np.stack(train_np, axis=-1)
    train_labels_np = npy_dict["labels"][:train_size]
    train_seq_len_np = npy_dict["seq_len"][:train_size]
    train_pos_tag_np = npy_dict["pos_tag_ids"][:train_size]
    train_eojeol_ids_np = npy_dict["eojeol_ids"][:train_size]
    print(f"train_np.shape: {train_np.shape}")
    print(f"train_labels_np.shape: {train_labels_np.shape}")
    print(f"train_seq_len_np.shape: {train_seq_len_np.shape}")
    print(f"train_pos_tag_ids_np.shape: {train_pos_tag_np.shape}")
    print(f"train_eojeol_ids_np.shape: {train_eojeol_ids_np.shape}")

    # dev
    dev_np = [npy_dict["input_ids"][train_size:valid_size],
              npy_dict["attention_mask"][train_size:valid_size],
              npy_dict["token_type_ids"][train_size:valid_size]]
    dev_np = np.stack(dev_np, axis=-1)
    dev_labels_np = npy_dict["labels"][train_size:valid_size]
    dev_seq_len_np = npy_dict["seq_len"][train_size:valid_size]
    dev_pos_tag_np = npy_dict["pos_tag_ids"][train_size:valid_size]
    dev_eojeol_ids_np = npy_dict["eojeol_ids"][train_size:valid_size]
    print(f"dev_np.shape: {dev_np.shape}")
    print(f"dev_labels_np.shape: {dev_labels_np.shape}")
    print(f"dev_seq_len_np.shape: {dev_seq_len_np.shape}")
    print(f"dev_pos_tag_ids_np.shape: {dev_pos_tag_np.shape}")
    print(f"dev_eojeol_ids_np.shape: {dev_eojeol_ids_np.shape}")

    # test
    test_np = [npy_dict["input_ids"][valid_size:],
               npy_dict["attention_mask"][valid_size:],
               npy_dict["token_type_ids"][valid_size:]]
    test_np = np.stack(test_np, axis=-1)
    test_labels_np = npy_dict["labels"][valid_size:]
    test_seq_len_np = npy_dict["seq_len"][valid_size:]
    test_pos_tag_np = npy_dict["pos_tag_ids"][valid_size:]
    test_eojeol_ids_np = npy_dict["eojeol_ids"][valid_size:]
    print(f"test_np.shape: {test_np.shape}")
    print(f"test_labels_np.shape: {test_labels_np.shape}")
    print(f"test_seq_len_np.shape: {test_seq_len_np.shape}")
    print(f"test_pos_tag_ids_np.shape: {test_pos_tag_np.shape}")
    print(f"test_eojeol_ids_np.shape: {test_eojeol_ids_np.shape}")

    root_path = "../data/npy/old_nikl/eojeol_electra"
    # save input_ids, attention_mask, token_type_ids
    np.save(root_path+"/train", train_np)
    np.save(root_path+"/dev", dev_np)
    np.save(root_path+"/test", test_np)

    # save labels
    np.save(root_path+"/train_labels", train_labels_np)
    np.save(root_path+"/dev_labels", dev_labels_np)
    np.save(root_path+"/test_labels", test_labels_np)

    # save seq_len
    np.save(root_path+"/train_seq_len", train_seq_len_np)
    np.save(root_path+"/dev_seq_len", dev_seq_len_np)
    np.save(root_path+"/test_seq_len", test_seq_len_np)

    # save pos_tag_ids
    np.save(root_path+"/train_pos_tag", train_pos_tag_np)
    np.save(root_path+"/dev_pos_tag", dev_pos_tag_np)
    np.save(root_path+"/test_pos_tag", test_pos_tag_np)

    # save eojeol_ids
    np.save(root_path+"/train_eojeol_ids", train_eojeol_ids_np)
    np.save(root_path+"/dev_eojeol_ids", dev_eojeol_ids_np)
    np.save(root_path+"/test_eojeol_ids", test_eojeol_ids_np)

    print(f"[make_gold_corpus_npy][save_eojeol_npy_dict] Complete - Save all npy files !")

### MAIN ###
if "__main__" == __name__:
    print("NPY MAKER !")

    all_sent_list = []
    with open("../data/pkl/ne_mp_old_nikl.pkl", mode="rb") as pkl_file:
        all_sent_list = pickle.load(pkl_file)
        print(f"[make_gold_corpus_npy][__main__] all_sent_list size: {len(all_sent_list)}")
    all_sent_list = conv_TTA_ne_category(all_sent_list)
    print(f"[make_gold_corpus_npy][__main__] all_sent_list.len: {len(all_sent_list)}") # 371,571

    # make npy
    #make_pos_tag_npy(tokenizer_name="klue/bert-base", src_list=all_sent_list, max_len=128)
    # make_wordpiece_npy(tokenizer_name="monologg/koelectra-base-v3-discriminator",
    #                    src_list=all_sent_list, max_len=128)
    make_eojeol_datasets_npy(tokenizer_name="monologg/koelectra-base-v3-discriminator",
                             src_list=all_sent_list, max_len=128, debug_mode=False)