import os
import logging
import torch.cuda
from dataclasses import dataclass

from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from transformers import ElectraForPreTraining, AdamW, get_linear_schedule_with_warmup
from fastprogress.fastprogress import master_bar, progress_bar

from Utils.data_def import TTA_NE_tags
from Utils.dataloder import ExoBrain_Datasets

@dataclass
class Argment:
    device: str = "cpu"
    model_name_or_path: str ="monologg/koelectra-base-v3-discriminator"
    num_labels: int = 0
    do_lower_case: bool = False
    do_train: bool = False
    do_eval: bool = False
    max_steps: int = -1
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 20
    warmup_proportion: int = 0
    max_grad_norm: float = 1.0
    train_batch_size: int = 32
    logging_steps: int = 1000
    save_steps: int = 1000
    save_optimizer: bool = False
    output_dir: str = "./"

####### logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)

####### train
def train(args, model, train_dataset, dev_dataset,
          learning_rate: int=0.01, ):
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
         'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
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
    mb = master_bar(range(int(args.num_train_epochs)))
    for epoch in mb:
        epoch_iterator = progress_bar(train_dataloader, parent=mb)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            inputs = {
                "input_ids": batch["input_ids"].to(args.device),
                "attention_mask": batch["attention_mask"].to(args.device),
                "token_type_ids": batch["token_type_ids"].to(args.device),
                "labels": batch["labels"].to(args.device)
            }
            outputs = model(**inputs)
            loss = outputs[0]

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
                global_step +=1

                # if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                #     if args.evaluate_test_during_training:
                #         evaluate(args, model, test_dataset, "test", global_step)
                #     else:
                #         evaluate(args, model, dev_dataset, "dev", global_step)

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )
                    model_to_save.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to {}".format(output_dir))

                    if args.save_optimizer:
                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to {}".format(output_dir))

            if args.max_steps > 0 and global_step > args.max_steps:
                break

        mb.write("Epoch {} done".format(epoch + 1))

    return global_step, tr_loss / global_step

def evaluate(args, model, eval_dataset, mode, global_step=None):
    results = {}
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset)

### MAIN ###
if "__main__" == __name__:
    print("[main.py][MAIN] -----MAIN")

    # Model
    arg = Argment()
    arg.device = "cuda" if torch.cuda.is_available() else "cpu"
    arg.model_name_or_path = "monologg/koelectra-base-v3-discriminator"
    arg.num_labels = len(TTA_NE_tags.keys())
    arg.do_train = True
    arg.train_batch_size = 8
    # arg.do_eval = True

    model = ElectraForPreTraining.from_pretrained(arg.model_name_or_path)
    model.to(arg.device)

    # load train dataset
    train_dataset = ExoBrain_Datasets(path="./datasets/exobrain/npy/ko-electra-base")
    dev_dataset = []
    test_dataset = []

    # do train
    if arg.do_train:
        global_step, tr_loss = train(arg, model, train_dataset, dev_dataset)
        logger.info(f"global_step = {global_step}, average loss = {tr_loss}")