import json
import os
import copy
from Utils.datasets_maker.nikl.data_def import NE_Json, NE
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
                    exo_ne_data = NE_Json(sent_id=sent_obj["id"],
                                          text=sent_obj["text"])
                    ne_arr = sent_obj["NE"]
                    for ne_obj in ne_arr:
                        exo_ne = NE(id=ne_obj["id"], text=ne_obj["text"],
                                    type=ne_obj["type"], begin=ne_obj["begin"], end=ne_obj["end"],
                                    weight=ne_obj["weight"], common_noun=ne_obj["common_noun"])
                        exo_ne_data.ne_list.append(copy.deepcopy(exo_ne))
                    # end, ne_arr loop
                # end, sent_arr loop
        #end, src_file_list loop
                    ret_list.append(copy.deepcopy(exo_ne_data))

        return ret_list

class NIKL_Parser:
    def __init__(self, dir_path: str=""):
        print("[data_parser][NIKL_Parser] ----Init")

        # check path
        print(f"[data_parser][NIKL_Parser] Check - path: {dir_path}")
        if not os.path.exists(dir_path):
            print(f"[data_parser][NIKL_Parser] ERR - Not Existed: {dir_path}")
            return
        else:
            self.dir_path = dir_path
            self.json_file_list = [dir_path+"/"+x for x in os.listdir(self.dir_path)]
            print(self.json_file_list)

    def parse_NE_data(self, src_path: str=""):
        ret_list = []

        # set target
        src_file_list = []
        if 0 < len(src_path):
            src_file_list.append(src_path)
        else:
            src_file_list = self.json_file_list

        # parsing
        for file_path in src_file_list:
            with open(file_path, mode="r", encoding="utf-8") as src_file:
                print(f"Parse: {file_path}....")
                root_obj = json.load(src_file)
                doc_arr = root_obj["document"]
                for d_idx, doc_obj in enumerate(doc_arr):
                    if 0 == (d_idx+1) % 100: print(f"Doc: {d_idx+1} Processing...")
                    sent_arr = doc_obj["sentence"]
                    for s_idx, sent_obj in enumerate(sent_arr):
                        if 0 == (s_idx+1) % 100: print(f"Sent: {s_idx+1} Processing...")
                        nikl_ne_data = NE_Json(sent_id=sent_obj["id"],
                                               text=sent_obj["form"])
                        ne_arr = sent_obj["ne"]
                        for ne_obj in ne_arr:
                            nikl_ne = NE(id=ne_obj["id"], text=ne_obj["form"],
                                         type=ne_obj["label"], begin=ne_obj["begin"], end=ne_obj["end"])
                            nikl_ne_data.ne_list.append(copy.deepcopy(nikl_ne))
                        ret_list.append(copy.deepcopy(nikl_ne_data))
                        # end, ne_arr
                    # end, sent_arr
                # end, doc_arr
        #end, src_file_list

        return ret_list

### MAIN ###
if "__main__" == __name__:
    print("[data_parser][MAIN] ----MAIN")

    # EXO Parser
    is_use_EXO_parser = False
    if is_use_EXO_parser:
        exo_parser = EXO_Parser(dir_path="../../../datasets/exobrain/news")
        exo_ne_list = exo_parser.parse_NE_data()

        # save_all ne_dataset
        save_path = "../../../datasets/exobrain/res_extract_ne/exo_ne_datasets.pkl"
        is_write_pkl = True
        if is_write_pkl:
            with open(save_path, mode="wb") as pk_file:
                pickle.dump(exo_ne_list, pk_file)

        is_load_pkl = False
        if is_load_pkl:
            with open(save_path, mode="rb") as pk_file:
                load_list = pickle.load(pk_file)
                print(len(load_list))

    # NIKL Parser
    is_use_NIKL_Parser = True
    if is_use_NIKL_Parser:
        nikl_parser = NIKL_Parser(dir_path="../../../datasets/NIKL/json")
        nikl_ne_list = nikl_parser.parse_NE_data()
        print(len(nikl_ne_list)) # 1,342,431

        save_path = "../../../datasets/NIKL/res_nikl_ne/nikl_ne_datasets.pkl"
        is_write_pkl = True
        if is_write_pkl:
            with open(save_path, mode="wb") as pk_file:
                pickle.dump(nikl_ne_list, pk_file)

        is_load_pkl = True
        if is_load_pkl:
            with open(save_path, mode="rb") as pk_file:
                load_list = pickle.load(pk_file)
                print(len(load_list))