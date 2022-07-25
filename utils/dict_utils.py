import copy
import re
import os
import pickle
import xml.etree.ElementTree as ET

from typing import List
from dataclasses import dataclass, field

### Data def
@dataclass
class Word_Info:
    word: str = ""
    word_unit: str = ""
    word_type: str = ""

@dataclass
class Sense_Info:
    sense_no: str = ""
    pos: str = ""
    type: str = ""
    definition: str = ""

@dataclass
class Dict_Item:
    target_code: int = field(init=False, default=0)
    word_info: Word_Info = field(init=False, default=Word_Info())
    sense_info: Sense_Info = field(init=False, default=Sense_Info())

#===============================================================
def read_korean_dict_xml(src_dir_path: str):
#===============================================================
    ret_kr_dict_item_list: List[Dict_Item] = []

    print(f"[dict_utils][read_korean_dict_xml] src_dir_path: {src_dir_path}")
    src_dict_list = [src_dir_path+"/"+path for path in os.listdir(src_dir_path)]
    print(f"[dict_utils][read_korean_dict_xml] size: {len(src_dict_list)},\n{src_dict_list}")

    '''
        @NOTE
            989450_1100000.xml 파일의 1474638~9 line에 parse 되지 않는 토큰 있음
            989450_550000.xml 파일의 1035958~9 line에 parse 되지 않는 토큰 있음
    '''
    for src_idx, src_dict_xml in enumerate(src_dict_list):
        print(f"[dict_utils][read_korean_dict_xml] {src_idx+1}, {src_dict_xml}")
        document = ET.parse(src_dict_xml)
        root = document.getroot() # <channel>

        for item_tag in root.iter(tag="item"):
            dict_item = Dict_Item()
            dict_item.target_code = int(item_tag.find("target_code").text)

            # word info
            word_info = Word_Info()
            word_info_tag = item_tag.find("wordInfo")
            word_info.word = word_info_tag.find("word").text
            word_info.word_unit = word_info_tag.find("word_unit").text
            word_info.word_type = word_info_tag.find("word_type").text

            # sense info
            sense_info = Sense_Info()
            sense_info_tag = item_tag.find("senseInfo")
            sense_info.sense_no = sense_info_tag.find("sense_no").text
            if "pos" in sense_info_tag.keys():
                sense_info.pos = sense_info_tag.find("pos").text
            sense_info.type = sense_info_tag.find("type").text
            sense_info.definition = sense_info_tag.find("definition").text

            dict_item.word_info = copy.deepcopy(word_info)
            dict_item.sense_info = copy.deepcopy(sense_info)

            ret_kr_dict_item_list.append(dict_item)

    print(f"[dict_utils][read_korean_dict_xml] total dict item size: {len(ret_kr_dict_item_list)}")
    return ret_kr_dict_item_list

### Main
if "__main__" == __name__:
    dir_path = "../data/dict/우리말샘"
    res_kr_dict_item_list = read_korean_dict_xml(src_dir_path=dir_path)

    # save *.pkl
    save_path = "../우리말샘_dict.pkl"
    with open(save_path, mode="wb") as save_pkl:
        pickle.dump(res_kr_dict_item_list, save_pkl)
        print(f"[dict_utils][__main__] Complete save - {save_path}")