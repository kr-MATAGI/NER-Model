from konlpy.tag import Mecab  # use mac
import MeCab # use mac

### MAIN ###
if "__main__" == __name__:
    print("[mecab_example.py][main] -- main")
    mecab = Mecab()
    mac_macb = MeCab.Tagger()

    test_word = "20~30"
    print(mecab.pos(test_word))
    print(mac_macb.parse(test_word))