import json
import os
import copy
from data_def import TTA_NE_Tags, EXO_NE_Json, EXO_NE
import pickle
from typing import List

### Class ###
class EXO_Parser:
    def __init__(self, dir_path: str=""):
        print("[data_parser][EXO_Parser] ----Init")

        # check path
        print(f"[data_parser][EXO_Parser] Check - path: {dir_path}")
        if not os.path.exists(dir_path):
            print(f"[data_parser][EXO_Parser] ERR - Not Existed: {dir_path}")
            return
        else:
            self.dir_path = dir_path
            self.json_file_list = [dir_path+"/"+x for x in os.listdir(self.dir_path)]
            print(self.json_file_list)

    def parse_NE_data(self, src_path: str="") -> List:
        ret_list = []

        # set target
        src_file_list = []
        if 0 < len(src_path):
            src_file_list.append(src_path)
        else:
            src_file_list = self.json_file_list

        # parsing
        for file_path in src_file_list:
            with open(file_path, mode="r", encoding="utf-8") as  src_file:
                root_obj = json.load(src_file)
                sent_arr = root_obj["sentence"]
                for sent_obj in sent_arr:
                    exo_ne_data = EXO_NE_Json(sent_id=sent_obj["id"],
                                              text=sent_obj["text"])
                    ne_arr = sent_obj["NE"]
                    for ne_obj in ne_arr:
                        exo_ne = EXO_NE(id=ne_obj["id"], text=ne_obj["text"],
                                        type=ne_obj["type"], begin=ne_obj["begin"], end=ne_obj["end"],
                                        weight=ne_obj["weight"], common_noun=ne_obj["common_noun"])
                        exo_ne_data.ne_list.append(copy.deepcopy(exo_ne))
                    # end, ne_arr loop
                # end, sent_arr loop
        #end, src_file_list loop
                    ret_list.append(copy.deepcopy(exo_ne_data))

        return ret_list

### MAIN ###
if "__main__" == __name__:
    print("[data_parser][MAIN] ----MAIN")

    exo_parser = EXO_Parser(dir_path="../datasets/exobrain/news")
    exo_ne_list = exo_parser.parse_NE_data()

    # save_all ne_dataset
    save_path = "../datasets/exobrain/res_extract_ne/exo_ne_datasets.pkl"
    is_write_pkl = False
    if is_write_pkl:
        with open(save_path, mode="wb") as pk_file:
            pickle.dump(exo_ne_list, pk_file)

    is_load_pkl = True
    if is_load_pkl:
        with open(save_path, mode="rb") as pk_file:
            load_list = pickle.load(pk_file)
            print(len(load_list))