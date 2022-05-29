import sentencepiece as spm
import pickle
from typing import List
from data_def import Sentence, Morp, NE, Word
from tokenizers import SentencePieceBPETokenizer
from transformers import AutoModel

def save_only_text(src_path: str, save_path: str):
    load_pkl_list: List[Sentence] = []
    with open(src_path, mode="rb") as load_pkl:
        load_pkl_list = pickle.load(load_pkl)

    only_text = []
    for idx, load_item in enumerate(load_pkl_list):
        if 0 == (idx % 10000):
            print(f"{idx} is Processsing... {load_item.text}")
        only_text.append(load_item.text)

    # save
    with open(save_path, "w", encoding="utf-8") as write_file:
        for txt in only_text:
            try:
                write_file.write(txt + "\n")
            except TypeError as TE:
                print(txt, TE)

### MAIN ###
if "__main__" == __name__:
    print("[sent_tokenizer][main] START")

    # save_only_text(src_path="../data/pkl/ne_mp_old_nikl.pkl",
    #                save_path="./sentpiece/nikl_text.txt")

    # vocab_size = 35000
    # corpus_file = "./sentpiece/nikl_text.txt"
    # min_frequency = 5
    # special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    #
    # tokenizer = SentencePieceBPETokenizer()
    # tokenizer.train(files=corpus_file, vocab_size=vocab_size,
    #                 min_frequency=min_frequency,
    #                 special_tokens=special_tokens,
    #                 show_progress=True)
    # tokenizer.save_model("./sentpiece", "sentpiece_model")

    tokenizer = SentencePieceBPETokenizer()
    tokenizer.load("./sentpiece", "sentpiece_model")

    a = tokenizer.encode("나는 최재훈 입니다.").ids
    print(a)
