from transformers import ElectraTokenizer

if "__main__" == __name__:
    tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

    test = "맞춤형 원천징수제도에 따라 원천징수세액을 80%로 선택한 근로자는 그간 낸 세금이 적어 연말정산 때 세금을 더 낼 가능성이 크다."
    tokens = tokenizer.tokenize(test)
    print(tokens)
    ne_list = ["맞춤형 원천징수제도", "원천징수세액", "80%", "근로자", "연말"]

    prev_end_idx = 0
    for ne in ne_list:
        cmp_word = ""
        is_ne = False
        begin_idx = 0
        end_idx = 0

        concat_ne = ne.replace(" ", "")
        for tdx, tok in enumerate(tokens):
            if tdx < prev_end_idx: continue

            if is_ne:
                cmp_word += tok.replace("##", "")
            elif tok in concat_ne:
                begin_idx = tdx
                cmp_word = tok
                is_ne = True

            if cmp_word == concat_ne:
                print(cmp_word)

                end_idx = tdx
                prev_end_idx = end_idx
                is_ne = False
                cmp_word = ""
                break