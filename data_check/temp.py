import pickle
import copy
from typing import List

from data_def import Sentence, Morp, NE, Word
from transformers import AutoTokenizer

### MAIN ##
if "__main__" == __name__:
    src_file_path = "../data/pkl/ne_mp_old_nikl.pkl"
    tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-small-v3-discriminator")
    results = { # tag: (word_item, morp_item_list) pair
        "JX": [],
        "EP": [],
        "EF": [],
        "EC": [],
        "ETM": [],
    }
    src_list: List[Sentence] = []
    with open(src_file_path, mode="rb") as load_pkl:
        src_list = pickle.load(load_pkl)
        print(f"LOAD SIZE: {len(src_list)}")

    target_pos_list = results.keys()
    end_point_pos = ["SF", "SP", "SS", "SE", "SO", "SW"]

    for src_idx, src_item in enumerate(src_list):
        if 0 == (src_idx % 1000):
            print(f"{src_idx} is Processing... {src_item.text}")

        new_morp_list: List[Morp] = []
        ignore_morp_id = []
        for morp_idx, morp_item in enumerate(src_item.morp_list):
            if morp_item.id in ignore_morp_id:
                continue
            if morp_item.label in target_pos_list:
                target_word = src_item.word_list[morp_item.word_id - 1]
                target_morp_list = [x for x in src_item.morp_list if x.word_id == morp_item.word_id]
                print(target_word, target_morp_list)

                # fix morp (use concat)
                new_label_form = copy.deepcopy(target_word.form)
                for tm_idx, target_morp_item in enumerate(target_morp_list):
                    if target_morp_item.form in target_word.form:
                        new_label_form = new_label_form.replace(target_morp_item.form, "")
                        if 0 >= len(new_label_form) and 2 <= len(target_morp_list):  # 만나/VV + 아/EC
                            vvec_label_form = target_morp_item.form
                            vvec_morp_label = target_morp_list[tm_idx:tm_idx + 3]
                            vvec_morp_label = [x.label for x in vvec_morp_label]
                            vvec_morp_label = "+".join(vvec_morp_label)
                            if "VV+EC" == vvec_morp_label or "VCP+EC" == vvec_morp_label or \
                                    "VCP+EF":
                                new_morp = Morp(id=morp_item.id, form=vvec_label_form,
                                                label=vvec_morp_label, word_id=morp_item.word_id,
                                                position=target_morp_list[tm_idx].position)
                                new_morp_list.append(target_morp_item)
                                ignore_morp_id.append(new_morp.id)
                                break
                        new_morp_list.append(target_morp_item)
                        ignore_morp_id.append(target_morp_item.id)
                    else:
                        end_idx = -1
                        for e_i in range(tm_idx, len(target_morp_list)):
                            if target_morp_list[e_i].label not in end_point_pos:
                                end_idx = e_i
                            else:
                                break
                        concat_morp_label = target_morp_list[tm_idx:end_idx + 1]
                        concat_morp_label = [x.label for x in concat_morp_label]
                        concat_morp_label = "+".join(concat_morp_label)
                        for e_i in range(end_idx, len(target_morp_list)):
                            new_label_form = new_label_form.replace(target_morp_list[e_i].form, "")
                        print(new_label_form, concat_morp_label)
                        new_morp = Morp(id=morp_item.id, form=new_label_form, label=concat_morp_label,
                                        word_id=morp_item.word_id, position=target_morp_list[tm_idx].position)
                        '''
                            # [Morp(id=41, form='받', label='VV', word_id=21, position=1),
                            # Morp(id=42, form='았', label='EP', word_id=21, position=2),
                            # Morp(id=43, form='다', label='EF', word_id=21, position=3)
                            2번 입력됨 -> ignore_morp_id
                        '''
                        # 살린다 -> 살리/VV + ㄴ다/EF 라면 '살리'는 이미 77line에서 삽입되어 있음.
                        filter_list = target_morp_list[tm_idx:end_idx + 1]
                        ignore_morp_id.append(new_morp.id)
                        print("FILTER:", filter_list)
                        new_morp_list = [x for x in new_morp_list if x not in filter_list]
                        new_morp_list.append(new_morp)
                        break
            else:
                new_morp_list.append(morp_item)
                ignore_morp_id.append(morp_item.id)
        print("A", src_item.morp_list)
        del_overlap_morp_list = []
        for new_morp_item in new_morp_list:
            if new_morp_item not in del_overlap_morp_list:
                del_overlap_morp_list.append(new_morp_item)
        src_list[src_idx].morp_list = del_overlap_morp_list
        print("B", src_item.morp_list)
        input()