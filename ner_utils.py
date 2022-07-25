import logging
import numpy as np
import torch
import random

from seqeval import metrics as seqeval_metrics
from sklearn import metrics as sklearn_metrics

# config, model
from transformers import ElectraConfig, AutoConfig
from model.electra_custom_model import ELECTRA_POS_LSTM
from model.custom_embed import Custom_Embed_Model
from model.bert_custom_model import (
    BERT_POS_LSTM, BERT_IDCNN_CRF
)

#===============================================================
def print_parameters(args, logger):
#===============================================================
    logger.info(f"ckpt_dir: {args.ckpt_dir}")
    logger.info(f"output_dir: {args.output_dir}")

    logger.info(f"train_npy: {args.train_npy}")
    logger.info(f"dev_npy: {args.dev_npy}")
    logger.info(f"test_npy: {args.test_npy}")

    logger.info(f"evaluate_test_during_training: {args.evaluate_test_during_training}")
    logger.info(f"eval_all_checkpoints: {args.eval_all_checkpoints}")

    logger.info(f"save_optimizer: {args.save_optimizer}")
    logger.info(f"do_lower_case: {args.do_lower_case}")

    logger.info(f"do_train: {args.do_train}")
    logger.info(f"do_eval: {args.do_eval}")

    logger.info(f"max_seq_len: {args.max_seq_len}")
    logger.info(f"num_train_epochs: {args.num_train_epochs}")

    logger.info(f"weight_decay: {args.weight_decay}")
    logger.info(f"gradient_accumulation_steps: {args.gradient_accumulation_steps}")

    logger.info(f"adam_epsilon: {args.adam_epsilon}")
    logger.info(f"warmup_proportion: {args.warmup_proportion}")

    logger.info(f"max_steps: {args.max_steps}")
    logger.info(f"max_grad_norm: {args.max_grad_norm}")
    logger.info(f"seed: {args.seed}")

    logger.info(f"model_name_or_path: {args.model_name_or_path}")
    logger.info(f"train_batch_size: {args.train_batch_size}")
    logger.info(f"eval_batch_size: {args.eval_batch_size}")
    logger.info(f"learning_rate: {args.learning_rate}")

    logger.info(f"logging_steps: {args.logging_steps}")
    logger.info(f"save_steps: {args.save_steps}")

#===============================================================
def load_corpus_npy_datasets(src_path: str, mode: str="train"):
#===============================================================
    dataset_npy = np.load(src_path)
    seq_len_npy = np.load("/".join(src_path.split("/")[:-1]) + "/" + mode + "_seq_len.npy")
    pos_tag_npy = np.load("/".join(src_path.split("/")[:-1]) + "/" + mode + "_pos_tag.npy")

    return dataset_npy, seq_len_npy, pos_tag_npy

#===============================================================
def init_logger():
# ===============================================================
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    log_formatter = logging.Formatter("%(asctime)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)

    return logger

#===============================================================
def set_seed(args):
#===============================================================
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if "cuda" == args.device:
        torch.cuda.manual_seed_all(args.seed)

#===============================================================
def f1_pre_rec(labels, preds, is_ner=True):
#===============================================================
    if is_ner:
        return {
            "precision": seqeval_metrics.precision_score(labels, preds),
            "recall": seqeval_metrics.recall_score(labels, preds),
            "f1": seqeval_metrics.f1_score(labels, preds),
        }
    else:
        return {
            "precision": sklearn_metrics.precision_score(labels, preds, average="macro"),
            "recall": sklearn_metrics.recall_score(labels, preds, average="macro"),
            "f1": sklearn_metrics.f1_score(labels, preds, average="macro"),
        }

#===============================================================
def show_ner_report(labels, preds):
#===============================================================
    return seqeval_metrics.classification_report(labels, preds)


#===============================================================
def load_ner_config_and_model(user_select: int, args, tag_dict):
#===============================================================
    config = None
    model = None

    # config
    if 1 == user_select:
        # ELECTRA+LSTM(POS)+CRF
        config = ElectraConfig.from_pretrained(args.model_name_or_path,
                                               num_labels=len(tag_dict.keys()),
                                               id2label={str(i): label for i, label in enumerate(tag_dict.keys())},
                                               label2id={label: i for i, label in enumerate(tag_dict.keys())})
        config.num_pos_labels = 49  # 국립국어원 형태 분석 말뭉치
    elif 2 == user_select:
        # BERT+LSTM(POS)+CRF
        config = AutoConfig.from_pretrained(args.model_name_or_path,
                                            num_labels=len(tag_dict.keys()),
                                            id2label={str(i): label for i, label in enumerate(tag_dict.keys())},
                                            label2id={label: i for i, label in enumerate(tag_dict.keys())})
        config.num_pos_labels = 49  # NIKL
    elif 3 == user_select:
        # BERT+IDCNN+CRF
        config = AutoConfig.from_pretrained(args.model_name_or_path,
                                            num_labels=len(tag_dict.keys()),
                                            id2label={str(i): label for i, label in enumerate(tag_dict.keys())},
                                            label2id={label: i for i, label in enumerate(tag_dict.keys())})
        # idcnn
        config.filter_nums = 64
        config.idcnn_nums = 4
        config.max_len = 128
    elif 4 == user_select:
        # CUSTOM EMBED MODEL
        config = AutoConfig.from_pretrained(args.model_name_or_path,
                                            num_labels=len(tag_dict.keys()),
                                            id2label={str(i): label for i, label in enumerate(tag_dict.keys())},
                                            label2id={label: i for i, label in enumerate(tag_dict.keys())})
        config.num_pos_labels = 49  # NIKL
        config.max_seq_len = 128

    # model
    if 1 == user_select:
        # ELECTRA+LSTM(POS)+CRF
        model = ELECTRA_POS_LSTM.from_pretrained(args.model_name_or_path, config=config)
    elif 2 == user_select:
        # BERT+LSTM(POS)+CRF
        model = BERT_POS_LSTM.from_pretrained(args.model_name_or_path, config=config)
    elif 3 == user_select:
        # BERT+IDCNN+CRF
        model = BERT_IDCNN_CRF(config=config)
    elif 4 == user_select:
        # CUSTOM EMBED MODEL
        model = Custom_Embed_Model(config=config)

    return config, model


#===============================================================
def load_model_checkpoints(user_select, checkpoint):
#===============================================================
    # model
    model = None
    if 1 == user_select:
        # ELECTRA+LSTM(POS)+CRF
        model = ELECTRA_POS_LSTM.from_pretrained(checkpoint)
    elif 2 == user_select:
        # BERT+LSTM(POS)+CRF
        model = BERT_POS_LSTM.from_pretrained(checkpoint)
    elif 3 == user_select:
        # BERT+IDCNN+CRF
        model = BERT_IDCNN_CRF.from_pretrained(checkpoint)

    return model