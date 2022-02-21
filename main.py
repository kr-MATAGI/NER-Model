import logging

from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import RandomSampler, DataLoader
from fastprogress.fastprogress import master_bar, progress_bar



# train
def train(model, train_dataset, dev_dataset,
          batch_size: int=16,
          learning_rate: int=0.01,):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * args.warmup_proportion),
                                                num_training_steps=t_total)

    global_step = 0
    tr_loss = 0.0

    model.zero_grad()
    mb = master_bar(range(int(arg.num_train_epochs)))
    for epoch in mb:
        epoch_iterator = progress_bar(train_dataloader, parent=mb)
        for step, batch in enumerate(epoch_iterator):
            model.train()

            outputs = model(**batch)
            loss = outputs[0]

            loss.backwar()
            tr_loss += loss.item()

            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1

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
