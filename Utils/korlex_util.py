import os
import sys
sys.path.append("C:/Users/MATAGI/Desktop/Git/De-identification-with-NER/Utils/KorLex_api") # windows
sys.path.append("C:/Users/MATAGI/Desktop/Git/De-identification-with-NER/Utils/datasets_maker/naver") # windows

#from konlpy.tag import Mecab # mac/linux
from eunjeon import Mecab # windows

import pickle
import copy

from Utils.KorLex_api.krx_api import KorLexAPI
from Utils.datasets_maker.naver import naver_def

## MeCab.pos.tags
nn_list = ["NNG", "NNP", "NNB", "NNBC", "NR", "NP"] # 명사
jk_list = ["JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC"] # 조사
ef_list = ["VCP+EF"] # 종결 어미
xs_list = ["XSN"] # 접미사
sx_list = ["SF", "SE", "SSO", "SSC", "SC"] # 마침표, 물음표, 느낌표, 줄임표, 여는/닫는 괄호, 구분자, 기타 기호

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

    # Init - Mecab
    mecab = Mecab()

    # load raw_data.pkl
    if not os.path.exists(src_path):
        print(f"[convert_korlex_naver_ner] ERR - Check src_path: {src_path}")
        return

    src_pkl_data = []
    with open(src_path, mode="rb") as src_pkl_file:
        src_pkl_data = pickle.load(src_pkl_file)
    print(f"[convert_korlex_naver_ner] src_pkl_data.len: {len(src_pkl_data)}")

    for pkl_idx, src_data in enumerate(src_pkl_data):
        src_sent = src_data.sentence
        src_word_list = src_data.word_list
        src_tag_list = src_data.tag_list

        assert len(src_word_list) == len(src_tag_list)

        del_idx_list = []
        src_iter = iter(zip(src_word_list, src_tag_list))
        for src_word, src_tag in src_iter:
            if "-" == src_tag:
                continue
            # 양 끝에 있는 특수기호 제거
            while True:
                res_mecab = mecab.pos(src_word)
                extract_sx_pos = list(filter(lambda x: x[-1] in sx_list, res_mecab))
                if (1 >= len(res_mecab)) or (0 >= len(extract_sx_pos)) or \
                        (len(extract_sx_pos) == len(res_mecab)):
                    break

                if res_mecab[0][-1] in sx_list:
                    del src_word_list[0]
                    del src_tag_list[0]
                elif res_mecab[-1][-1] in sx_list:
                    del src_word_list[-1]
                    del src_tag_list[-1]


### MAIN ###
if "__main__" == __name__:
    print("[korlex_util.py] \t Start")

    src_path = "../datasets/Naver_NLP/raw_data.pkl"
    save_path = "../datasets/Naver_NLP/korlex_data.pkl"

    convert_korlex_naver_ner(src_path=src_path, save_path=save_path)