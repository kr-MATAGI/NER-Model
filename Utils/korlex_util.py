import os
import sys

sys.path.append("/Users/matagi/Desktop/Git/De-identification-with-NER/Utils/KorLex_api") # mac
sys.path.append("/Users/matagi/Desktop/Git/De-identification-with-NER/Utils/datasets_maker/naver") # mac

sys.path.append("C:/Users/MATAGI/Desktop/Git/De-identification-with-NER/Utils/KorLex_api") # windows
sys.path.append("C:/Users/MATAGI/Desktop/Git/De-identification-with-NER/Utils/datasets_maker/naver") # windows

from konlpy.tag import Mecab # mac/linux
#from eunjeon import Mecab # windows

import pickle
import copy
from typing import List

from Utils.KorLex_api.krx_api import KorLexAPI
from Utils.datasets_maker.naver.naver_def import NAVER_NE

## MeCab.pos.tags
nn_list = ["NNG", "NNP", "NNB", "NNBC", "NR", "NP"] # 명사
jk_list = ["JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC"] # 조사
ef_list = ["VCP+EF"] # 종결 어미
xs_list = ["XSN"] # 접미사 (use ?)
# @Note : Mecab은 SO(붙임표), SW(수학, 화폐기호)를 분리하지 않고 SY를 사용한다.
sx_list = ["SF", "SE", "SSO", "SSC", "SC", "SY"] # 마침표, 물음표, 느낌표, 줄임표, 여는/닫는 괄호, 구분자, 기타 기호(수학 기호, 화폐 기호, 붙침표)

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

def remove_back_jk_pos(src_data: NAVER_NE) -> NAVER_NE:
    ret_naver_ne = copy.deepcopy(src_data)

    # Init - MeCab
    mecab = Mecab()

    mecab_target_tag = ["PER", "DATE", "TIME", "NUM"]  # 조사/접미사/종결 어미
    target_pos = jk_list + ef_list
    for idx, (word, tag) in enumerate(zip(src_data.word_list, src_data.tag_list)):
        conv_tag = tag.split("_")[0]

        if conv_tag in mecab_target_tag:
            res_mecab = mecab.pos(word)
            filter_target = list(filter(lambda x: x[-1] in target_pos, res_mecab))

            # 조사/종결 어미 제거
            conv_word = ""
            if 1 <= len(filter_target) and 2 <= len(res_mecab) and (len(filter_target) != len(res_mecab)):
                if res_mecab[-1][-1] in target_pos:
                    conv_word = word.replace(res_mecab[-1][0], "")
                    ret_naver_ne.word_list[idx] = conv_word
             
            # 접미사 제거
            res_mecab = mecab.pos(conv_word)
            filter_xsn = list(filter(lambda x: x[-1] in xs_list, res_mecab))

            if 1 <= len(filter_xsn) and 2 <= len(res_mecab) and (len(filter_xsn) != len(res_mecab)):
                if res_mecab[-1][-1] in xs_list:
                    conv_word = conv_word.replace(res_mecab[-1][0], "")
                    ret_naver_ne.word_list[idx] = conv_word

    return ret_naver_ne

def remove_back_jk_by_krx(krx_api, src_data: NAVER_NE) -> NAVER_NE:
    ret_naver_ne = copy.deepcopy(src_data)

    assert len(src_data.word_list) == len(src_data.tag_list)

    # Init - MeCab
    mecab = Mecab()

    '''
        @Note : korlex에서 명사 품사의 단어로 검색이 되면 뒤 해당 단어가 나올 때까지 붙은 문자/단어 제거
                그렇지 않으면 뒤에 조사/특수문자 하나만 제거
        @Example : 수강생 -> Mecab, [(수강, NNG), (생, XSN), (으로, JKB)]
    '''
    korlex_target_tag = ["LOC", "ORG", "AFW", "CVL", "TRM", "EVT", "ANM", "PLT", "MAT", "FLD"]
    target_pos = jk_list + ef_list
    list_len = len(src_data.word_list)
    for idx, (word, tag) in enumerate(zip(src_data.word_list, src_data.tag_list)):
        split_tag = tag.split("_")

        if split_tag[0] not in korlex_target_tag:
            continue
        if (idx != list_len-1) and ("B" == split_tag[1]) and ("I" == src_data.tag_list[idx+1].split("_")[-1]):
            continue

        # 1. 조사 / 종결 어미 제거
        res_mecab = mecab.pos(word)
        filter_pos = list(filter(lambda x: x[-1] in target_pos, res_mecab))
        if 1 <= len(filter_pos) and 2 <= len(res_mecab) and (len(filter_pos) != len(res_mecab)):
            # 1) 단어 전체가 korlex에서 명사로써 검색이 되는지
            res_krx = krx_api.search_word(word=word, ontology="KORLEX")
            if 1 <= len(res_krx):
                if "n" == res_krx[0][0].target.pos:
                    continue

            # 2) 앞 부분부터 명사구만 concat
            nn_word = ""
            for mecab_word in res_mecab:
                if mecab_word[-1] not in nn_list:
                    break
                nn_word += mecab_word[0]

            # 3) 앞 부분부터 추가해서 검색이 안될 때까지 concat
            krx_word = ""
            for mecab_word in res_mecab:
                temp_word = krx_word + mecab_word[0]
                res_krx = krx_api.search_word(word=temp_word, ontology="KORLEX")
                if 0 >= len(res_krx):
                    break
                if "n" != res_krx[0][0].target.pos:
                    break
                krx_word = temp_word

            # 4) 위의 2와 3의 단어 길이로 점수 비교하여 긴 것을 사용
            select_word = nn_word if len(nn_word) >= len(krx_word) else krx_word
            
            # 5) empty한 단어가 선택되었을 경우, 조사와 접미사 제거
            if 0 >= len(select_word):
                except_target_pos = jk_list + xs_list + sx_list
                except_case = list(filter(lambda x: x[-1] not in except_target_pos, res_mecab))
                if 1 <= len(except_case):
                    ret_naver_ne.word_list[idx] = "".join([x[0] for x in except_case])
                    print(res_mecab, "->", ret_naver_ne.word_list[idx]) # for TEST
            else:
                ret_naver_ne.word_list[idx] = select_word




    return ret_naver_ne


def convert_korlex_naver_ner(src_path: str, save_path: str) -> None:
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

    # Do Filter
    new_pkl_data = []
    for pkl_idx, src_data in enumerate(src_pkl_data):
        if 0 == ((pkl_idx+1) % 1000):
            print(f"{pkl_idx+1} Processing...")
        
        # 앞/뒤 특수 기호, 괄호, 구분자 제거
        res_sx_remove = remove_sx_pos(src_data)
        res_back_jk = remove_back_jk_pos(res_sx_remove)
        res_korlex = remove_back_jk_by_krx(krx_json_api, src_data=res_back_jk)
        new_pkl_data.append(res_korlex)
    print(f"results.size : {len(new_pkl_data)}")

    # save
    with open(save_path, mode="wb") as save_pkl:
        pickle.dump(new_pkl_data, save_pkl)
        print(f"Complete Save - {save_path}")

### MAIN ###
if "__main__" == __name__:
    print("[korlex_util.py] \t Start")

    src_path = "../datasets/Naver_NLP/raw_data.pkl"
    save_path = "../datasets/Naver_NLP/korlex_data.pkl"

    convert_korlex_naver_ner(src_path=src_path, save_path=save_path)