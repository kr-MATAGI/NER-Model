import pickle
import os
import copy

import sys
sys.path.append("/Users/matagi/Desktop/Git/De-identification-with-NER/Utils/datasets_maker/naver") # mac

# tagger
#from konlpy.tag import Mecab # mac/linux
from eunjeon import Mecab # windows
from Utils.datasets_maker.naver.naver_def import NAVER_NE

nn_list = ["NNG", "NNP", "NNB", "NNBC", "NR", "NP"] # 명사
jk_list = ["JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC"] # 조사
ef_list = ["VCP+EF"] # 종결 어미
xs_list = ["XSN"] # 접미사
sx_list = ["SF", "SE", "SSO", "SSC", "SC", "SY"] # 마침표, 물음표, 느낌표, 줄임표, 여는/닫는 괄호, 구분자, 기타 기호

def filter_fn(pos_pair):
    return pos_pair[-1] in (jk_list + ef_list + xs_list + sx_list)

def conv_jk_pos_tag(pos_tags) -> list:
    if (pos_tags[-1][1] in jk_list) and (pos_tags[-2][1] in nn_list):
        pos_tags.remove(pos_tags[-1])
    return copy.deepcopy(pos_tags)

def check_nikl_morphs(src_path: str) -> None:
    # check src_path
    if not os.path.exists(src_path):
        print("[mecab_util.py][check_nikl_morphs] Check - src_path,", src_path)
        return

    # load *.pkl
    src_datasets = []
    with open(src_path, mode="rb") as src_file:
        src_datasets = pickle.load(src_file)
        print(f"[mecab_util.py][check_nikl_morphs] src_datasets size: {len(src_datasets)}")

    # check JK*, JX, JC
    mecab = Mecab()
    for ne_json in src_datasets:
        for ne_obj in ne_json.ne_list:
            pos_tags = mecab.pos(ne_obj.text)
            jk_tags = list(filter(filter_fn, pos_tags))

            if 0 < len(jk_tags):
                print(f"{ne_json.text}\n{ne_obj}\n{pos_tags}\n------")

def convert_mecab_nikl(src_path: str, save_path: str) -> None:
    # check src path
    if not os.path.exists(src_path):
        print(f"[mecab_util.py][convert_mecab_nikl] ERR - src_path: {src_path}")
        return

    # load *.pkl
    src_datasets = []
    with open(src_path, mode="rb") as src_file:
        src_datasets = pickle.load(src_file)
        print(f"[mecab_util.py][convert_mecab_nikl] src_datasets size: {len(src_datasets)}")

    # remove jk pos tag from PS, DT, TI, QT
    new_ne_json = []
    target_obj_type = ["PS", "DT", "TI", "QT"]
    mecab = Mecab()
    for ne_json in src_datasets:
        for ne_obj in ne_json.ne_list:
            ne_obj_type = ne_obj.type.split("_")[0]
            if ne_obj_type not in target_obj_type: # pass
                continue

            pos_tags = mecab.pos(ne_obj.text)
            filter_tags = list(filter(filter_fn, pos_tags))
            if 0 >= len(filter_tags): # pass
                continue

            if 1 < len(pos_tags):
                pos_tags = conv_jk_pos_tag(pos_tags)
                conv_text = "".join([item[0] for item in pos_tags])
                ne_obj.text = conv_text
        new_ne_json.append(copy.deepcopy(ne_json))

    # compare len
    print(f"[mecab_util.py][convert_mecab_nikl] len - [origin: {len(src_datasets)}, remake: {len(new_ne_json)}]")

    # save
    with open(save_path, mode="wb") as save_file:
        pickle.dump(new_ne_json, save_file)
        print("[mecab_util.py][convert_mecab_nikl] Complete - Save:", save_path)

def remove_sx_list(word, mecab):
    ret_word = copy.deepcopy(word)

    morphs = mecab.pos(ret_word)
    for mp in morphs:
        if mp[-1] in sx_list:
            ret_word = ret_word.replace(mp[0], "")

    return ret_word

def convert_mecab_naver(src_path: str, save_path: str) -> None:
    # check src path
    if not os.path.exists(src_path):
        print(f"[mecab_util.py][convert_mecab_naver] ERR - src_path: {src_path}")
        return

    # load *.pkl
    src_datasets = []
    with open(src_path, mode="rb") as src_file:
        src_datasets = pickle.load(src_file)
        print(f"[mecab_util.py][convert_mecab_naver] src_dataset size: {len(src_datasets)}")

    # use mecab
    new_ne_datasets = []
    NN_target_type = ["PER", "DATE", "TIME", "NUM"] # remove - jk_list, ef_list
    uniq_NN_target_type = ["LOC", "ORG", "AFW", "CVL", "TRM", "EVT", "ANM", "PLT", "MAT", "FLD"] # remove - jk_list, xs_list

    mecab = Mecab()
    for naver_ne in src_datasets:
        new_sentence = copy.deepcopy(naver_ne.sentence)
        new_word_list = []

        origin_data_len = len(naver_ne.word_list)
        for idx in range(origin_data_len):
            tag = naver_ne.tag_list[idx].split("_")[0]
            word = naver_ne.word_list[idx]

            if tag == "-":
                new_word_list.append(word)
                continue

            # remove - sx_list
            conv_sx_word = remove_sx_list(word, mecab)
            if 0 < len(conv_sx_word):
                new_sentence = naver_ne.sentence.replace(word, conv_sx_word)
            else:
                conv_sx_word = word

            if tag not in (NN_target_type + uniq_NN_target_type):
                new_word_list.append(copy.deepcopy(conv_sx_word))
            else:
                pos_tags = mecab.pos(conv_sx_word)
                filter_tags = list(filter(filter_fn, pos_tags))
                if 0 < len(filter_tags):
                    new_word = copy.deepcopy(conv_sx_word)
                    if tag in NN_target_type:
                        while True:
                            if 1 >= len(pos_tags):
                                break
                            if pos_tags[-1][-1] not in (jk_list + ef_list):
                                break

                            new_word = new_word.replace(pos_tags[-1][0], "")
                            pos_tags = pos_tags[:-1]

                    elif tag in uniq_NN_target_type:
                            if (pos_tags[-1][-1] in (jk_list + xs_list + ef_list)) and \
                                    (1 < len(new_word)):
                                new_word = new_word.replace(pos_tags[-1][0], "")
                                pos_tags = pos_tags[:-1]
                    new_word_list.append(new_word)
                    new_sentence = new_sentence.replace(word, new_word)
                else:
                    new_word_list.append(conv_sx_word)

        # For Check before/after
        print("", naver_ne.sentence, "\n", new_sentence)
        print("", naver_ne.word_list, "\n", new_word_list, len(new_word_list))
        print("", naver_ne.tag_list, len(naver_ne.tag_list), "\n")

        # append to new datasets list
        new_naver_ne = NAVER_NE(sentence=new_sentence,
                                word_list=new_word_list,
                                tag_list=naver_ne.tag_list)
        new_ne_datasets.append(copy.deepcopy(new_naver_ne))

    print(f"compare length - before: {len(src_datasets)}, after: {len(new_ne_datasets)}")

    # save save_path.pkl
    with open(save_path, mode="wb") as save_pkl:
        pickle.dump(new_ne_datasets, save_pkl)
        print("save complete !")

### MAIN ###
if "__main__" == __name__:
    mecab = Mecab()
    print(mecab.pos("12.6m의"))
    exit()

    print("[mecab_util.py][main] - start")

    src_path = "../datasets/Naver_NLP/raw_data.pkl"
    save_path = "../datasets/Naver_NLP/mecab_data.pkl"

    # NIKL
    # check method
    is_check_jk_method = False
    if is_check_jk_method:
        check_nikl_morphs(src_path=src_path)

    # remove jk pos tag at PS, DT, TI, QT
    is_mecab_nikl = False
    if is_mecab_nikl:
        convert_mecab_nikl(src_path=src_path, save_path=save_path)

    # Naver
    is_naver_nikl = True
    if is_naver_nikl:
        convert_mecab_naver(src_path=src_path, save_path=save_path)