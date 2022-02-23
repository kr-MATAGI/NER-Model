import pickle
import matplotlib.pyplot as plt
import numpy as np

import sys

from h5py.h5t import convert

sys.path.append("C:/Users/MATAGI/Desktop/Git/De-identification-with-NER/Utils")
from Utils.data_def import TTA_NE_tags


def convert_NE_tag(ne_tag: str = ""):
    ret_ne_tag = ne_tag

    if ("LCP" in ret_ne_tag) or ("LCG" in ret_ne_tag):  # location
        ret_ne_tag = "LC"
    elif "OGG" in ret_ne_tag:  # organization
        ret_ne_tag = "OG"
    elif ("AFW" in ret_ne_tag) or ("AFA" in ret_ne_tag):  # artifacts
        ret_ne_tag = "AF"
    elif ("TMM" in ret_ne_tag) or ("TMI" in ret_ne_tag) or \
            ("TMIG" in ret_ne_tag):  # term
        ret_ne_tag = "TM"

    return ret_ne_tag

### MAIN ##
if "__main__" == __name__:
    data_path = "../../datasets/NIKL/res_nikl_ne/nikl_ne_datasets.pkl"

    # count dict
    count_dict = {
        "PS": 0, "LC": 0, "OG": 0, "AF": 0, "DT": 0, "TI": 0,
        "CV": 0, "AM": 0, "PT": 0, "QT": 0, "FD": 0, "TR": 0,
        "EV": 0, "MT": 0, "TM": 0
    }
    print(f"tag size: {len(count_dict.keys())}")

    with open(data_path, mode="rb") as data_file:
        ne_data_list = pickle.load(data_file)

        for ne_json in ne_data_list:
            for ne in ne_json.ne_list:
                ne_tag = convert_NE_tag(ne.type.split("_")[0])
                count_dict[ne_tag] += 1

    print(f"result: \n{count_dict}")

    # plot
    tag_name = count_dict.keys()
    tag_count = [count_dict[k] for k in count_dict.keys()]

    x = np.arange(15)
    plt.bar(x, tag_count)
    plt.xlabel("tag name")
    plt.ylabel("count")
    plt.xticks(x, tag_name)
    plt.show()