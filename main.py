import copy
import os
import logging
import numpy as np
import torch.cuda
from dataclasses import dataclass

from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import ElectraConfig, get_linear_schedule_with_warmup

from tqdm import tqdm

from seqeval import metrics as seqeval_metrics
from sklearn import metrics as sklearn_metrics

from Utils.datasets_maker.nikl.data_def import TTA_NE_tags
from Utils.dataloder import NE_Datasets

from Utils.datasets_maker.naver.naver_def import NAVER_NE_MAP

from electra_crf_ner import ElectraCRF_NER


@dataclass
class Argment:
    is_load_model: bool = False
    device: str = "cpu"
    model_name_or_path: str = "monologg/koelectra-base-v3-discriminator"
    num_labels: int = 0
    do_lower_case: bool = False
    do_train: bool = False
    do_eval: bool = False
    do_test: bool = False
    evaluate_test_during_training: bool = False
    max_steps: int = -1
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 20
    warmup_proportion: int = 0
    max_grad_norm: float = 1.0
    train_batch_size: int = 32
    eval_batch_size: int = 32
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    logging_steps: int = 1000
    save_steps: int = 20000
    save_optimizer: bool = False
    output_dir: str = "./"
    n_gpu: int = 1


####### logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)

####### tensorboard
if not os.path.exists("./logs"):
    os.mkdir("./logs")
tb_writer = SummaryWriter("./logs")


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


####### train
def train(args, model, train_dataset, dev_dataset, test_dataset):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size)
    t_total = 0
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    logger.info(f"t_total: {t_total}")

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    # @NOTE: optimizer에 설정된 learning_rate까지 선형으로 감소시킨다. (스케줄러)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * args.warmup_proportion),
                                                num_training_steps=t_total)
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # Train9!
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

    model.zero_grad()
    for epoch in range(args.num_train_epochs):
        pbar = tqdm(train_dataloader)
        for step, batch in enumerate(pbar):
            model.train()
            inputs = {
                "input_ids": batch["input_ids"].to(args.device),
                "attention_mask": batch["attention_mask"].to(args.device),
                "token_type_ids": batch["token_type_ids"].to(args.device),
                "labels": batch["labels"].to(args.device)
            }

            outputs = model(**inputs)
            loss = outputs[0]

            # if 1 < args.n_gpu:
            # loss = loss.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss = loss.mean()
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

        logger.info("  Epoch Done= %d", epoch)
        pbar.close()

        evaluate(args, model, dev_dataset, "dev", global_step, epoch)

        # save samples
        if not os.path.exists("samples"):
            os.mkdir("samples")
        torch.save(model, "./model/epoch_{}.pt".format(epoch))

    return global_step, tr_loss / global_step


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
                "labels": batch["labels"].to(args.device)
            }
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()

        nb_eval_steps += 1
        tb_writer.add_scalar("Loss/val_" + str(train_epoch), eval_loss / nb_eval_steps, nb_eval_steps)
        eval_pbar.set_description("Eval Loss - %.04f" % (eval_loss / nb_eval_steps))

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    logger.info("  Eval End !")
    eval_pbar.close()

    eval_loss = eval_loss / nb_eval_steps
    results = {
        "loss": eval_loss
    }
    preds = np.argmax(preds, axis=2)

    # nikl
    #labels = TTA_NE_tags.keys()

    # naver
    naver_label_map = copy.deepcopy(NAVER_NE_MAP.keys())
    del naver_label_map["X"]
    labels = naver_label_map.keys()

    label_map = {i: label for i, label in enumerate(labels)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    result = f1_pre_rec(out_label_list, preds_list)
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


### MAIN ###
if "__main__" == __name__:
    print("[main.py][MAIN] -----MAIN")

    # arg
    args = Argment()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.model_name_or_path = "monologg/koelectra-small-v3-discriminator"

    # nikl
    #args.num_labels = len(TTA_NE_tags.keys())

    # naver
    args.num_labels = len(NAVER_NE_MAP.keys()) - 1 # except "X"

    args.num_train_epochs = 20
    args.train_batch_size = 32
    args.eval_batch_size = 32
    args.learning_rate = 5e-5

    args.evaluate_test_during_training = False
    args.save_optimizer = True
    args.save_steps = 2500
    args.weight_decay = 0.01

    # config
    # nikl
    # config = ElectraConfig.from_pretrained(args.model_name_or_path,
    #                                        num_labels=len(TTA_NE_tags.keys()),
    #                                        id2label={str(i): label for i, label in enumerate(TTA_NE_tags.keys())},
    #                                        label2id={label: i for i, label in enumerate(TTA_NE_tags.keys())})

    # naver
    config = ElectraConfig.from_pretrained(args.model_name_or_path,
                                           num_labels=len(NAVER_NE_MAP.keys()),
                                           id2label={str(i): label for i, label in enumerate(NAVER_NE_MAP.keys())},
                                           label2id={label: i for i, label in enumerate(NAVER_NE_MAP.keys())})

    # models
    args.is_load_model = False
    if args.is_load_model:
        model = torch.load("./filtered_model.pt")
    else:
        model = ElectraCRF_NER.from_pretrained(args.model_name_or_path, config=config)

    if 1 < torch.cuda.device_count():
        logging.info(f"Let's use {torch.cuda.device_count()} GPUs!")
        args.n_gpu = torch.cuda.device_count()
        model = torch.nn.DataParallel(model)
    model.to(args.device)

    # load train dataset
    train_dataset = NE_Datasets(path="./datasets/Naver_NLP/npy/mecab/train")
    dev_dataset = NE_Datasets(path="./datasets/Naver_NLP/npy/mecab/test")
    test_dataset = NE_Datasets(path="./datasets/Naver_NLP/npy/mecab/test")

    # do train
    args.do_train = True
    if args.do_train:
        global_step, tr_loss = train(args, model, train_dataset, dev_dataset, test_dataset)
        logger.info(f"global_step = {global_step}, average loss = {tr_loss}")

    args.do_test = False
    if args.do_test:
        results = evaluate(args, model, test_dataset, mode="test")

    # tensorboard close
    tb_writer.close()
