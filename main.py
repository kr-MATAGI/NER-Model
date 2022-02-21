from transformers import AutoModel


### MAIN ###
if "__main__" == __name__:
    print("[main.py][MAIN] -----MAIN")
    model = AutoModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
