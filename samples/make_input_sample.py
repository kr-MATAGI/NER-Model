from transformers import ElectraTokenizer

if "__main__" == __name__:
    tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

    test = "며칠전 고다(고양이라서 다행이야) 카페에 들어갔다가 재밌는 사진을 발견했다."
    tokens = tokenizer.tokenize(test)
    print(tokens)

    ne_list = ["고다", "고양이라서 다행이야"]

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