import copy
import json
import os
import logging
import numpy as np
import random

import glob
import re
import argparse
from attrdict import AttrDict

import torch
import torch.autograd as autograd
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoConfig, get_linear_schedule_with_warmup
from electra_custom_model import ELECTRA_LSTM_LAN

from tqdm import tqdm

from seqeval import metrics as seqeval_metrics
from sklearn import metrics as sklearn_metrics

## TAG SET
ETRI_TAG = {
    "O": 0,
    "B-PS": 1, "I-PS": 2,
    "B-LC": 3, "I-LC": 4,
    "B-OG": 5, "I-OG": 6,
    "B-AF": 7, "I-AF": 8,
    "B-DT": 9, "I-DT": 10,
    "B-TI": 11, "I-TI": 12,
    "B-CV": 13, "I-CV": 14,
    "B-AM": 15, "I-AM": 16,
    "B-PT": 17, "I-PT": 18,
    "B-QT": 19, "I-QT": 20,
    "B-FD": 21, "I-FD": 22,
    "B-TR": 23, "I-TR": 24,
    "B-EV": 25, "I-EV": 26,
    "B-MT": 27, "I-MT": 28,
    "B-TM": 29, "I-TM": 30
}


#===============================================================
def init_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    log_formatter = logging.Formatter("%(asctime)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)

    return logger

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if "cuda" == args.device:
        torch.cuda.manual_seed_all(args.seed)

def f1_pre_rec(labels, preds, is_ner=True):
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

def show_ner_report(labels, preds):
    return seqeval_metrics.classification_report(labels, preds)

#===============================================================
class Char_NER_Dataset(Dataset):
    def __init__(self, data: np.ndarray, seq_len: np.ndarray, num_labels: int):
        self.num_labels = num_labels

        self.input_ids = data[:][:, :, 0]
        self.labels = data[:][:, :, 1]
        self.attention_mask = data[:][:, :, 2]
        self.token_type_ids = data[:][:, :, 3]
        self.input_seq_len = seq_len

        self.input_ids = torch.tensor(self.input_ids, dtype=torch.long)
        self.attention_mask = torch.tensor(self.attention_mask, dtype=torch.long)
        self.token_type_ids = torch.tensor(self.token_type_ids, dtype=torch.long)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        self.input_seq_len = torch.tensor(self.input_seq_len, dtype=torch.long)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        input_label_seq_tensor = autograd.Variable(torch.zeros(self.num_labels), volatile=False).long()
        input_label_seq_tensor[:self.num_labels] = torch.LongTensor([i for i in range(self.num_labels)])
        items = {
            "attention_mask": self.attention_mask[idx],
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
            "token_type_ids": self.token_type_ids[idx],
            "input_seq_len": self.input_seq_len[idx],
            "input_label_seq_tensor": input_label_seq_tensor
        }

        return items

#===============================================================
def evaluate(args, model, eval_dataset, mode, global_step=None, train_epoch=0):
    results = {}

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval
    if None != global_step:
        logger.info("***** Running evaluation on {} dataset ({} step) *****".format(mode, global_step))
    else:
        logger.info("***** Running evaluation on {} dataset *****".format(mode))

    logger.info("  Num examples = {}".format(len(eval_dataset)))
    logger.info("  Eval Batch size = {}".format(args.eval_batch_size))
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    eval_pbar = tqdm(eval_dataloader)
    for batch in eval_pbar:
        model.eval()
        with torch.no_grad():
            inputs = {
                "input_ids": batch["input_ids"].to(args.device),
                "attention_mask": batch["attention_mask"].to(args.device),
                "token_type_ids": batch["token_type_ids"].to(args.device),
                "input_seq_len": batch["input_seq_len"].to(args.device),
                "labels": batch["labels"].to(args.device),
                "input_label_seq_tensor": batch["input_label_seq_tensor"].to(args.device) # [batch_size, num_labels]
            }

            loss, seq_tag = model(**inputs)
            if 2 <= args.n_gpu:
                loss = loss.mean()
            eval_loss += loss

        nb_eval_steps += 1
        tb_writer.add_scalar("Loss/val_" + str(train_epoch), eval_loss / nb_eval_steps, nb_eval_steps)
        eval_pbar.set_description("Eval Loss - %.04f" % (eval_loss / nb_eval_steps))

        if preds is None:
            preds = seq_tag.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, seq_tag.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    logger.info("  Eval End !")
    eval_pbar.close()

    eval_loss = eval_loss / nb_eval_steps
    results = {
        "loss": eval_loss
    }

    labels = ETRI_TAG.keys()
    label_map = {i: label for i, label in enumerate(labels)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    ignore_index = torch.nn.CrossEntropyLoss().ignore_index
    ignore_list = [ignore_index, ETRI_TAG["O"]]
    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] not in ignore_list:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    result = f1_pre_rec(out_label_list, preds_list, is_ner=True)
    results.update(result)

    output_dir = os.path.join(args.output_dir, mode)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_eval_file = os.path.join(output_dir,
                                    "{}-{}.txt".format(mode, global_step) if global_step else "{}.txt".format(mode))
    with open(output_eval_file, "w") as f_w:
        logger.info("***** Eval results on {} dataset *****".format(mode))
        for key in sorted(results.keys()):
            logger.info("  {} = {}".format(key, str(results[key])))
            f_w.write("  {} = {}\n".format(key, str(results[key])))
        logger.info("\n" + show_ner_report(out_label_list, preds_list))  # Show report for each tag result
        f_w.write("\n" + show_ner_report(out_label_list, preds_list))

    return results

#===============================================================
def train(args, model, train_dataset, dev_dataset):
    train_data_len = len(train_dataset)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (train_data_len // args.gradient_accumulation_steps) + 1
    else:
        t_total = (train_data_len // args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # eps : 줄이기 전/후의 lr차이가 eps보다 작으면 무시한다.
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # @NOTE: optimizer에 설정된 learning_rate까지 선형으로 감소시킨다. (스케줄러)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * args.warmup_proportion),
                                                num_training_steps=t_total)

    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Logging steps = %d", args.logging_steps)
    logger.info("  Save steps = %d", args.save_steps)

    global_step = 0
    tr_loss = 0.0

    train_sampler = RandomSampler(train_dataset)

    model.zero_grad()
    for epoch in range(args.num_train_epochs):
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
        pbar = tqdm(train_dataloader)
        for step, batch in enumerate(pbar):
            model.train()
            inputs = {
                "input_ids": batch["input_ids"].to(args.device),
                "attention_mask": batch["attention_mask"].to(args.device),
                "token_type_ids": batch["token_type_ids"].to(args.device),
                "input_seq_len": batch["input_seq_len"].to(args.device),
                "labels": batch["labels"].to(args.device),
                "input_label_seq_tensor": batch["input_label_seq_tensor"].to(args.device) # [batch_size, num_labels]
            }

            outputs = model(**inputs)
            loss = outputs[0]

            if 2 <= args.n_gpu:
                loss = loss.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0 or \
                (len(train_dataloader) <= args.gradient_accumulation_steps and \
                 (step + 1) == len(train_dataloader)
                ):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                tb_writer.add_scalar("Loss/train", tr_loss / global_step, global_step)
                pbar.set_description("Train Loss - %.04f" % (tr_loss / global_step))

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    evaluate(args, model, dev_dataset, "dev", global_step)

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save samples checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )
                    model_to_save.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving samples checkpoint to {}".format(output_dir))

                    if args.save_optimizer:
                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to {}".format(output_dir))

            if args.max_steps > 0 and global_step > args.max_steps:
                break

        logger.info("  Epoch Done= %d", epoch + 1)
        pbar.close()

    return global_step, tr_loss / global_step

#===============================================================
# Logger
logger = init_logger()

# Tensorboard
if not os.path.exists("./logs"):
    os.mkdir("./logs")
tb_writer = SummaryWriter("./logs")

def main(cli_args):
    with open(cli_args.config_file) as config_file:
        args = AttrDict(json.load(config_file))
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.n_gpu = torch.cuda.device_count() # check gpu count

    set_seed(args)

    logger.info(f"Training/Evaluation parameters {args}")
    args.output_dir = os.path.join(args.ckpt_dir, args.output_dir)

    # Config
    config = AutoConfig.from_pretrained(args.model_name_or_path,
                                        num_labels=len(ETRI_TAG.keys()),
                                        id2label={str(i): label for i, label in enumerate(ETRI_TAG.keys())},
                                        label2id={label: i for i, label in enumerate(ETRI_TAG.keys())})
    config.max_seq_len = 128 # for label_embedding

    model = ELECTRA_LSTM_LAN(args.model_name_or_path, config=config,
                             is_use_gru=args.is_gru, is_use_high_way=args.is_highway)

    # GPU or CPU
    if 1 < torch.cuda.device_count():
        logging.info(f"Let's use {torch.cuda.device_count()} GPUs!")
        args.n_gpu = torch.cuda.device_count()
        model = torch.nn.DataParallel(model)
    model.to(args.device)

    train_dataset = np.load(args.train_npy)
    dev_dataset = np.load(args.dev_npy)
    test_dataset = np.load(args.test_npy)
    print(f"train_dataset.shape: {train_dataset.shape}")
    print(f"dev_dataset.shape: {dev_dataset.shape}")
    print(f"test_dataset.shape: {test_dataset.shape}")

    train_seq_len = np.load("/".join(args.train_npy.split("/")[:-1]) + "/train_seq_len.npy")
    valid_seq_len = np.load("/".join(args.dev_npy.split("/")[:-1]) + "/dev_seq_len.npy")
    test_seq_len = np.load("/".join(args.train_npy.split("/")[:-1]) + "/test_seq_len.npy")
    print(f"train_seq_len.shape: {train_seq_len.shape}")
    print(f"valid_seq_len.shape: {valid_seq_len.shape}")
    print(f"test_seq_len.shape: {test_seq_len.shape}")

    train_dataset = Char_NER_Dataset(data=train_dataset, seq_len=train_seq_len, num_labels=config.num_labels)
    dev_dataset = Char_NER_Dataset(data=dev_dataset, seq_len=valid_seq_len, num_labels=config.num_labels)
    test_dataset = Char_NER_Dataset(data=test_dataset, seq_len=test_seq_len, num_labels=config.num_labels)

    if args.do_train:
        global_step, tr_loss = train(args, model, train_dataset, dev_dataset)
        logger.info(f"global_step = {global_step}, average loss = {tr_loss}")

    results = {}
    if args.do_eval:
        checkpoints = list(os.path.dirname(c) for c in
                           sorted(glob.glob(args.output_dir + "/**/" + "pytorch_model.bin", recursive=True),
                                  key=lambda path_with_step: list(map(int, re.findall(r"\d+", path_with_step)))[-1]))

        if not args.eval_all_checkpoints:
            checkpoints = checkpoints[-1:]
        else:
            logging.getLogger("transformers.configuration_utils").setLevel(logging.WARN)  # Reduce logging
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1]
            model = ELECTRA_LSTM_LAN.from_pretrained(checkpoint, model_name=args.model_name_or_path, config=config,
                                                     is_use_gru=args.is_gru, is_use_high_way=args.is_highway)
            model.to(args.device)
            result = evaluate(args, model, test_dataset, mode="test", global_step=global_step)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as f_w:
            if len(checkpoints) > 1:
                for key in sorted(results.keys(), key=lambda key_with_step: (
                        "".join(re.findall(r'[^_]+_', key_with_step)),
                        int(re.findall(r"_\d+", key_with_step)[-1][1:])
                )):
                    f_w.write("{} = {}\n".format(key, str(results[key])))
            else:
                for key in sorted(results.keys()):
                    f_w.write("{} = {}\n".format(key, str(results[key])))


#===============================================================
if "__main__" == __name__:
    cli_parser = argparse.ArgumentParser()

    cli_parser.add_argument("--config_file", type=str, required=True)
    cli_args = cli_parser.parse_args()

    main(cli_args)

    # Tensorboard close
    tb_writer.close()