import pickle
import logging


def init_logger() -> logging.Logger:
    logger = logging.getLogger("[data_filter.py]")
    logger.setLevel(logging.INFO)

    # handler 객체 생성
    stream_handler = logging.StreamHandler()

    # formatter 객체 생성
    formatter = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # handler에 level 설정
    stream_handler.setLevel(logging.INFO)

    # handler에 format 설정
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)

    return logger

def filter_NIKL_datasets(src_path: str, save_path: str) -> None:
    logger = init_logger()

    origin_size = 0
    filtered_size = 0
    new_datasets = []
    with open(src_path, mode="rb") as src_file:
        ne_pkl_list = pickle.load(src_file)

        origin_size = len(ne_pkl_list)
        for ne_data in ne_pkl_list:
            is_exist_ne = False
            if 0 < len(ne_data.ne_list):
                is_exist_ne = True

            is_pass_sent_len = False
            if 10 < len(ne_data.text):
                is_pass_sent_len = True

            if all([is_exist_ne, is_pass_sent_len]):
                new_datasets.append(ne_data)
    filtered_size = len(new_datasets)

    logger.info(f"origin_size: {origin_size}")
    logger.info(f"filtered_size: {filtered_size}")
    logger.info(f"diff_size: {origin_size - filtered_size}")

    with open(save_path, mode="wb") as save_file:
        pickle.dump(new_datasets, save_file)

### MAIN ###
if "__main__" == __name__:
    print("[data_filter.py] ----MAIN")

    filter_NIKL = True
    if filter_NIKL:
        filter_NIKL_datasets(src_path="../datasets/NIKL/res_nikl_ne/nikl_ne_datasets.pkl",
                             save_path="../datasets/NIKL/res_nikl_ne/filtered_nikl_ne_datasets.pkl")