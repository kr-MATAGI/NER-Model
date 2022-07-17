from transformers import ElectraTokenizer

### TEST ###
if "__main__" == __name__:
    test_str = ["[PAD]", "[SEP]", "[MASK]", "[CLS]", "[UNK]"]

    tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    res = tokenizer.convert_tokens_to_ids(test_str)

    print(res)