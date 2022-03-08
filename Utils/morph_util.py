import pickle
import os
import copy

# tagger
from konlpy.tag import Mecab

nn_list = ["NNG", "NNP", "NNB", "NNBC", "NR", "NP"]
jk_list = ["JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC"]
def find_jk(pos_pair):
    return pos_pair[-1] in jk_list

def conv_jk_pos_tag(pos_tags) -> list:
    if (pos_tags[-1][1] in jk_list) and (pos_tags[-2][1] in nn_list):
        pos_tags.remove(pos_tags[-1])
    return copy.deepcopy(pos_tags)

def check_jk_at_datasets(src_path: str) -> None:
    # check src_path
    if not os.path.exists(src_path):
        print("[morph_util.py][check_jk_at_datasets] Check - src_path,", src_path)
        return

    # load *.pkl
    src_datasets = []
    with open(src_path, mode="rb") as src_file:
        src_datasets = pickle.load(src_file)
        print(f"[morph_util.py][check_jk_at_datasets] src_datasets size: {len(src_datasets)}")

    # check JK*, JX, JC
    mecab = Mecab()
    for ne_json in src_datasets:
        for ne_obj in ne_json.ne_list:
            pos_tags = mecab.pos(ne_obj.text)
            jk_tags = list(filter(find_jk, pos_tags))

            if 0 < len(jk_tags):
                print(f"{ne_json.text}\n{ne_obj}\n{pos_tags}\n------")

def remove_jk_pos_tag(src_path: str, save_path: str) -> None:
    # check src path
    if not os.path.exists(src_path):
        print(f"[morph_util.py][remove_jk_pos_tag] ERR - src_path: {src_path}")
        return

    # load *.pkl
    src_datasets = []
    with open(src_path, mode="rb") as src_file:
        src_datasets = pickle.load(src_file)
        print(f"[morph_util.py][remove_jk_pos_tag] src_datasets size: {len(src_datasets)}")

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
            filter_tags = list(filter(find_jk, pos_tags))
            if 0 >= len(filter_tags): # pass
                continue

            if 1 < len(pos_tags):
                pos_tags = conv_jk_pos_tag(pos_tags)
                conv_text = "".join([item[0] for item in pos_tags])
                ne_obj.text = conv_text
        new_ne_json.append(copy.deepcopy(ne_json))

    # compare len
    print(f"[morph_util.py][remove_jk_pos_tag] len - [origin: {len(src_datasets)}, remake: {len(new_ne_json)}]")

    # save
    with open(save_path, mode="wb") as save_file:
        pickle.dump(new_ne_json, save_file)
        print("[morph_util.py][remove_jk_pos_tag] Complete - Save:", save_path)

### MAIN ###
if "__main__" == __name__:
    print("[morph_util.py][main] - start")

    src_path = "../datasets/NIKL/res_nikl_ne/filtered_nikl_ne_datasets.pkl"
    save_path = "../datasets/NIKL/res_nikl_ne/mecab_jk_nikl_datasets.pkl"

    # check method
    is_check_jk_method = False
    if is_check_jk_method:
        check_jk_at_datasets(src_path=src_path)

    # remove jk pos tag at PS, DT, TI, QT
    is_remove_jk_pos_tag = True
    if is_remove_jk_pos_tag:
        remove_jk_pos_tag(src_path=src_path, save_path=save_path)