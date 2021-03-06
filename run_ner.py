import json
import os
import platform
import pickle
import numpy as np

import glob
import re
from attrdict import AttrDict

import torch
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer

from tqdm import tqdm

### Model
from ner_def import (
    ETRI_TAG, NER_MODEL_LIST,
)
from ner_datasets import NER_POS_Dataset
from ner_utils import (
    init_logger, print_parameters, load_corpus_npy_datasets,
    set_seed, show_ner_report, f1_pre_rec, load_ner_config_and_model,
    load_model_checkpoints
)

### Global variable
g_user_select = 0
logger = init_logger()

if not os.path.exists("./logs"):
    os.mkdir("./logs")
tb_writer = SummaryWriter("./logs")

# Dictionary
dict_hash_table = None

### Evaluate
#===============================================================
def evaluate(args, model, eval_dataset, mode, global_step=None, train_epoch=0):
#===============================================================
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
                "labels": batch["labels"].to(args.device),
                "input_seq_len": batch["input_seq_len"].to(args.device),
                "pos_tag_ids": batch["pos_tag_ids"].to(args.device),
                "span_ids": batch["span_ids"].to(args.device)
            }

            log_likelihood, outputs = model(**inputs)
            #loss, logits = outputs[:2]
            loss = -1 * log_likelihood

            # outputs = model(**inputs)
            # loss = outputs.loss
            eval_loss += loss.mean().item()

        nb_eval_steps += 1
        tb_writer.add_scalar("Loss/val_" + str(train_epoch), eval_loss / nb_eval_steps, nb_eval_steps)
        eval_pbar.set_description("Eval Loss - %.04f" % (eval_loss / nb_eval_steps))

        if preds is None:
            preds = np.array(outputs)
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, np.array(outputs), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        # if preds is None:
        #     preds = outputs.logits.detach().cpu().numpy()
        #     out_label_ids = inputs["labels"].detach().cpu().numpy()
        # else:
        #     preds = np.append(preds, outputs.logits.detach().cpu().numpy(), axis=0)
        #     out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    logger.info("  Eval End !")
    eval_pbar.close()

    eval_loss = eval_loss / nb_eval_steps
    results = {
        "loss": eval_loss
    }

    # 07.25
    # preds = np.argmax(preds, axis=2)

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


### Train
#===============================================================
def train(args, model, train_dataset, dev_dataset):
#===============================================================
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

    # eps : ????????? ???/?????? lr????????? eps?????? ????????? ????????????.
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # @NOTE: optimizer??? ????????? learning_rate?????? ???????????? ???????????????. (????????????)
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
                "labels": batch["labels"].to(args.device),
                "input_seq_len": batch["input_seq_len"].to(args.device),
                "pos_tag_ids": batch["pos_tag_ids"].to(args.device),
                "span_ids": batch["span_ids"].to(args.device)
            }

            # inputs["input_ids"].shape -> [batch_size, max_seq_len]

            log_likelihood, outputs = model(**inputs)
            loss = -1 * log_likelihood
            # outputs = model(**inputs)
            # loss = outputs.loss

            if 1 < args.n_gpu:
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
def main():
#===============================================================
    # Model Select
    print("=======================================")
    print("Please select model (default: 1):")
    print(f"1. {NER_MODEL_LIST[0]}")
    print(f"2. {NER_MODEL_LIST[1]}")
    print(f"3. {NER_MODEL_LIST[2]}")
    print(f"4. {NER_MODEL_LIST[3]}")
    print(f"5. {NER_MODEL_LIST[4]}")
    print("=======================================")
    print(">>>> number: ")

    global g_user_select
    g_user_select = int(input())
    config_file_path = "./config/electra-pos-tag.json"

    if g_user_select > len(NER_MODEL_LIST.keys()):
        g_user_select = 1
    if 1 == g_user_select:
        config_file_path = "./config/electra-pos-tag.json"
    elif 2 == g_user_select:
        config_file_path = "./config/bert-pos-tag.json"
    elif 3 == g_user_select:
        config_file_path = "./config/bert-idcnn-crf.json"
    elif 4 == g_user_select:
        config_file_path = "./config/custom-embed-model.json"
    elif 5 == g_user_select:
        config_file_path = "./config/electra-custom_embed-model.json"

    with open(config_file_path) as config_file:
        args = AttrDict(json.load(config_file))
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if "Darwin" == platform.system() and torch.backends.mps.is_built():
        args.device = "mps" if torch.backends.mps.is_available() else "cpu"

    set_seed(args)

    args.output_dir = os.path.join(args.ckpt_dir, args.output_dir)

    # Config
    config, model = load_ner_config_and_model(g_user_select, args, ETRI_TAG)

    # print config / model
    logger.info(f"[run_ner][__main__] model: {args.model_name_or_path}")
    logger.info(f"Training/Evaluation parameters")
    print_parameters(args, logger)

    # GPU or CPU
    if 1 < torch.cuda.device_count():
        logger.info(f"Let's use {torch.cuda.device_count()} GPUs!")
        args.n_gpu = torch.cuda.device_count()
        model = torch.nn.DataParallel(model)
    model.to(args.device)

    # load train/dev/test npy
    train_dataset, train_seq_len, train_pos_tag, train_span_ids = load_corpus_npy_datasets(args.train_npy, mode="train")
    dev_dataset, dev_seq_len, dev_pos_tag, dev_span_ids = load_corpus_npy_datasets(args.dev_npy, mode="dev")
    test_dataset, test_seq_len, test_pos_tag, test_span_ids = load_corpus_npy_datasets(args.test_npy, mode="test")
    print(f"train.shape - dataset: {train_dataset.shape}, seq_len: {train_seq_len.shape}, "
          f"pos_tag: {train_pos_tag.shape}, span_ids: {train_span_ids.shape}")
    print(f"dev.shape - dataset: {dev_dataset.shape}, seq_len: {dev_seq_len.shape}, "
          f"pos_tag: {dev_pos_tag.shape}, span_ids: {dev_span_ids.shape}")
    print(f"test.shape - dataset: {test_dataset.shape}, seq_len: {test_seq_len.shape}, "
          f"pos_tag: {test_pos_tag.shape}, span_ids: {test_span_ids.shape}")

    # make train/dev/test dataset
    train_dataset = NER_POS_Dataset(data=train_dataset, seq_len=train_seq_len,
                                    pos_data=train_pos_tag, span_ids=train_span_ids)
    dev_dataset = NER_POS_Dataset(data=dev_dataset, seq_len=dev_seq_len,
                                  pos_data=dev_pos_tag, span_ids=dev_span_ids)
    test_dataset = NER_POS_Dataset(data=test_dataset, seq_len=test_seq_len,
                                   pos_data=test_pos_tag, span_ids=test_span_ids)

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
            logger.info("transformers.configuration_utils")
            logger.info("transformers.modeling_utils")
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1]
            model = load_model_checkpoints(g_user_select, checkpoint)
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


### Code Main
if "__main__" == __name__:
    main()

    # close tensorboard
    tb_writer.close()