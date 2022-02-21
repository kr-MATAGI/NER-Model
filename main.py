from transformers import AutoModel
import logging


### MAIN ###
if "__main__" == __name__:
    print("[main.py][MAIN] -----MAIN")

    # logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_formatter = logging.Formatter('%(asctime)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)

    # Model
    model = AutoModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
