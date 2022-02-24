from transformers import ElectraTokenizer

### TEST ###
if "__main__" == __name__:
    test_str = "'13월의 보너스냐 세금폭탄이냐' 연말정산서비스 15일 시작."

    tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    res = tokenizer.tokenize(test_str)
    print(res)

    print(5e-5 == 0.00005)