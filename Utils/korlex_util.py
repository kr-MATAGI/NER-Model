import os
import sys
sys.path.append("C:/Users/MATAGI/Desktop/Git/De-identification-with-NER/Utils/KorLex_api") # windows
sys.path.append("C:/Users/MATAGI/Desktop/Git/De-identification-with-NER/Utils/datasets_maker/naver") # windows

#from konlpy.tag import Mecab # mac/linux
from eunjeon import Mecab # windows

import pickle
import copy
from typing import List

from Utils.KorLex_api.krx_api import KorLexAPI
from Utils.datasets_maker.naver.naver_def import NAVER_NE

## MeCab.pos.tags
nn_list = ["NNG", "NNP", "NNB", "NNBC", "NR", "NP"] # 명사
jk_list = ["JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC"] # 조사
ef_list = ["VCP+EF"] # 종결 어미
xs_list = ["XSN"] # 접미사
sx_list = ["SF", "SE", "SSO", "SSC", "SC"] # 마침표, 물음표, 느낌표, 줄임표, 여는/닫는 괄호, 구분자

### Method
def remove_sx_pos(src_data: NAVER_NE) -> NAVER_NE:
    ret_naver_ne = copy.deepcopy(src_data)

    # Init - Mecab
    mecab = Mecab()

    while True:
        assert len(ret_naver_ne.word_list) == len(ret_naver_ne.tag_list)
        list_len = len(ret_naver_ne.word_list)

        is_change = False
        for idx in range(list_len):
            split_tag = ret_naver_ne.tag_list[idx].split("_") # e.g. ORG_B -> ORG, I

            if "-" == split_tag[-1]:
                if idx == (list_len - 1):
                    is_break = True
                else:
                    continue

            # ignore B - I
            if "B" == split_tag[0] and idx != (list_len-1) and "I" == ret_naver_ne.tag_list[idx+1].split("_")[0]:
                continue

            res_mecab = mecab.pos(ret_naver_ne.word_list[idx])
            sx_filter = list(filter(lambda x: x[-1] in sx_list, res_mecab))

            if (1 <= len(sx_filter)) and (2 <= len(res_mecab)) and \
                    ((res_mecab[0][-1] in sx_list) or (res_mecab[-1][-1] in sx_list)):
                if res_mecab[0][-1] in sx_list:
                    # front character
                    del_ch = ret_naver_ne.word_list[idx][0]
                    ret_naver_ne.word_list[idx] = ret_naver_ne.word_list[idx].replace(del_ch, "")
                    is_change = True
                elif res_mecab[-1][-1] in sx_list:
                    # back character
                    del_ch = ret_naver_ne.word_list[idx][-1]
                    ret_naver_ne.word_list[idx] = ret_naver_ne.word_list[idx].replace(del_ch, "")
                    is_change = True
        if not is_change:
            break

    return ret_naver_ne

def convert_korlex_naver_ner(src_path: str, save_path: str) -> None:
    mecab_target_tag = ["PER", "DATE", "TIME", "NUM"] # 조사/접미사/종결 어미
    '''
        @Note : korlex에서 명사 품사의 단어로 검색이 되면 뒤 해당 단어가 나올 때까지 붙은 문자/단어 제거
                그렇지 않으면 뒤에 조사/특수문자 하나만 제거
    '''
    korlex_target_tag = ["LOC", "ORG", "AFW", "CVL", "TRM", "EVT", "ANM", "PLT", "MAT", "FLD"]

    # Init - KorLex
    krx_json_api = KorLexAPI(ssInfo_path="../Utils/KorLex_api/dic/korlex_ssInfo.pkl",
                             seIdx_path="../Utils/KorLex_api/dic/korlex_seIdx.pkl",
                             reIdx_path="../Utils/KorLex_api/dic/korlex_reIdx.pkl")
    krx_json_api.load_synset_data()

    # load raw_data.pkl
    if not os.path.exists(src_path):
        print(f"[convert_korlex_naver_ner] ERR - Check src_path: {src_path}")
        return

    src_pkl_data = []
    with open(src_path, mode="rb") as src_pkl_file:
        src_pkl_data = pickle.load(src_pkl_file)
    print(f"[convert_korlex_naver_ner] src_pkl_data.len: {len(src_pkl_data)}")

    for pkl_idx, src_data in enumerate(src_pkl_data):
        if 0 == ((pkl_idx+1) % 1000):
            print(f"{pkl_idx+1} Processing...")
        res_sx_remove = remove_sx_pos(src_data)



### MAIN ###
if "__main__" == __name__:
    print("[korlex_util.py] \t Start")

    src_path = "../datasets/Naver_NLP/raw_data.pkl"
    save_path = "../datasets/Naver_NLP/korlex_data.pkl"

    convert_korlex_naver_ner(src_path=src_path, save_path=save_path)