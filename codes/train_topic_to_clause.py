import os
import time
import random
import json
from functools import partial
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BartConfig, BartTokenizerFast, BartForConditionalGeneration
from transformers import get_linear_schedule_with_warmup
from accelerate import Accelerator

import wandb
from nltk.translate.bleu_score import corpus_bleu
from icecream import ic

from kwd_to_clause_dataset import KwdToClauseDataset, collate_top_to_clause


def train_epoch(train_dataloader, dev_dataloader, tokenizer, model, optimizer, scheduler, 
    accelerator, save_dict, logger):
    
    model.train()
    train_info = save_dict['train_info']
    curr_epoch = train_info['curr_epoch']
    step_loss = 0.0
    total_loss = 0.0
    n_steps = len(train_dataloader)
    start_time = time.time()
    optimizer.zero_grad()
    
    for i, sample in enumerate(train_dataloader):
        with accelerator.autocast():
            op = model(**sample)
            loss = op.loss
        
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), 2.0, norm_type=2)

        if (i+1) % train_info['accumulate_train_batches'] == 0 or (i+1) == n_steps:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        step_loss += loss.item()
        total_loss += loss.item()

        if accelerator.is_local_main_process:
            lr = optimizer.param_groups[0]['lr']
            if (i+1) % train_info['log_steps'] == 0:
                cur_loss = step_loss / train_info['log_steps']
                ms_per_batch = (time.time() - start_time) * 1000 / train_info['log_steps']
                
                train_info['curr_step'] = i+1
                train_info['avg_step_train_losses'].append(cur_loss)
                train_info['avg_ms_per_batch'].append(ms_per_batch)

                accelerator.print(f'| epoch {curr_epoch:3d} | step {i+1:5d} / {n_steps:5d} batches '
                    f'| milli-sec/batch {ms_per_batch:7.2f} | loss {cur_loss:7.2f} |')

                logger.log({'lr': lr, 'train/step/#': (i+1), 'train/step/loss': cur_loss, 
                    'train/step/ms_per_batch': ms_per_batch})

                step_loss = 0.0
                start_time = time.time()

        if (i+1) % train_info['validate_every_n_steps'] == 0:
            dev_start = time.time()
            dev_loss, dev_bleu = evaluate(dev_dataloader, tokenizer, model, accelerator, 
                dev_bleu_every_k_random_steps=train_info['dev_bleu_every_k_random_steps'])
            dev_loss = accelerator.gather(dev_loss).mean().cpu().item()
            dev_bleu = accelerator.gather(dev_bleu).mean().cpu().item()
                        
            if accelerator.is_local_main_process:
                if dev_bleu > train_info['best_dev_bleu']:
                    train_info['best_dev_loss'] = dev_bleu
                    save_checkpoint('best.pth', model, optimizer, accelerator, save_dict, 
                        store_optimizer_state=False)
                    accelerator.print('*'*10, 'Updated as best checkpoint', '*'*10)
                    
                dev_duration = time.time() - dev_start
                train_info['dev_bleus'].append(dev_bleu)
                train_info['avg_dev_losses'].append(dev_loss)
                train_info['dev_durations'].append(dev_duration)
                
                accelerator.print(f'|| epoch {curr_epoch:3d} | dev bleu {dev_bleu:7.2f} ' +
                    f'| dev duration {dev_duration:7.2f} sec | dev loss {dev_loss:5.2f} ||')

                logger.log({'dev/duration': dev_duration, 'dev/bleu': dev_bleu, 
                    'dev/loss': dev_loss})
    lr = optimizer.param_groups[0]['lr']
    
    return total_loss / n_steps, lr


@torch.no_grad()
def evaluate(dev_dataloader, tokenizer, model, accelerator, dev_bleu_every_k_random_steps=1):
    model.eval()
    all_ops = list()
    all_refs = list()
    losses = 0.0
    
    for sample in dev_dataloader:
        op = model(**sample)
        loss = op.loss

        loss_mean = accelerator.gather(loss).mean()
        losses += loss_mean.item()

        i = random.randint(1, dev_bleu_every_k_random_steps)
        if i == dev_bleu_every_k_random_steps:
            if type(model) == torch.nn.parallel.distributed.DistributedDataParallel:
                module = model.module
            else:
                module = model
            
            ops = tokenizer.batch_decode(module.generate(sample['input_ids']), 
                skip_special_tokens=True)
            all_ops.extend(ops)
            
            ref_ids = deepcopy(sample['labels'])
            ref_ids[ref_ids == -100] = 1
            refs = tokenizer.batch_decode(ref_ids, skip_special_tokens=True)
            all_refs.extend(refs)

    losses /= len(dev_dataloader)
    try:
        bleu = corpus_bleu([[r] for r in all_refs], all_ops) * 100.0
    except ZeroDivisionError:
        print('Zero division error caught at BLEU score computation')
        bleu = 0.0
    losses = torch.tensor([losses], dtype=torch.float32).to(accelerator.device)
    bleu = torch.tensor([bleu], dtype=torch.float32).to(accelerator.device)
    
    return losses, bleu


def save_checkpoint(filename, model, optimizer, accelerator, save_dict, store_optimizer_state=False):
    os.makedirs(save_dict['experiment_name'], exist_ok=True)
    
    unwrapped_model = accelerator.unwrap_model(model)
    save_dict['model_state_dict'] = unwrapped_model.state_dict()
    if store_optimizer_state:
        unwrapped_optimizer = accelerator.unwrap_model(optimizer)
        save_dict['optimizer_state_dict'] = unwrapped_optimizer.state_dict()
    accelerator.save(save_dict, os.path.join(save_dict['experiment_name'], filename))
    
    return


def train(train_dataloader, dev_dataloader, tokenizer, model, optimizer, scheduler, accelerator, save_dict, 
    logger):
    
    train_info = save_dict['train_info']
    for epoch in range(1, train_info['total_epochs']+1):
        epoch_start_time = time.time()
        train_info['curr_epoch'] = epoch
        
        train_start_time = time.time()
        train_loss, lr = train_epoch(train_dataloader, dev_dataloader, tokenizer, model, 
            optimizer, scheduler, accelerator, save_dict, logger)
        train_duration = time.time() - train_start_time

        accelerator.wait_for_everyone()
        accelerator.print(f'Training complete for epoch {epoch} with average loss: {train_loss}, ' + 
            f'current learning rate {lr}')
        accelerator.print('Starting validation')
        
        if accelerator.is_local_main_process:
            logger.log({'train/epoch/duration': train_duration, 'train/epoch/loss': train_loss})
            
            epoch_duration = time.time() - epoch_start_time
            train_info['epoch_durations'].append(epoch_duration)
            save_checkpoint(f'epoch-{epoch}.pth', model, optimizer, accelerator, save_dict, 
                store_optimizer_state=False)
        
            accelerator.print('-'*90)
            accelerator.print(f'| end of epoch {epoch:3d} | time: {epoch_duration:7.2f}s | ' + 
                f'train loss {train_loss:5.2f} |')
            accelerator.print('-'*90)
            logger.log({'epoch/#': epoch, 'epoch/duration': epoch_duration})

    return


if __name__ == '__main__':
    accelerator = Accelerator()

    TRAIN_CLAUSE_KWDS = 'topic-nostep-clause-kwds.c100.k200.lemstop.train.json'
    DEV_CLAUSE_KWDS = 'topic-nostep-clause-kwds.c100.k200.lemstop.dev.json'
    TEST_CLAUSE_KWDS = 'topic-nostep-clause-kwds.c100.k200.lemstop.test.json'

    CLAUSE_KWD_LIMIT = 10
    INPUT_MAX_LENGTH = 25
    OUTPUT_MAX_LENGTH = 700

    TRAIN_BATCH_SIZE = 4
    EVAL_BATCH_SIZE = 8
    ACCUMULATE_TRAIN_BATCHES = 8
    NUM_DATALOADER_PROCESSES = 1
    N_EPOCHS = 15
    LOG_STEPS = 200
    VALIDATE_PER_EPOCH = 0.5
    DEV_BLEU_EVERY_K_RANDOM_STEPS = 10
    EXP_NAME = 'top-to-clause'

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    OPTIMIZER_ARGS = {
        'lr': 1e-5,
        'betas': (0.9, 0.98),
        'weight_decay': 1e-2,
        'eps': 1e-9
    }

    PRETRAINED_MODEL_NAME = 'facebook/bart-base'

    with open(TRAIN_CLAUSE_KWDS, 'r') as f:
        train_clause_kwds = json.load(f)
    with open(DEV_CLAUSE_KWDS, 'r') as f:
        dev_clause_kwds = json.load(f)
    with open(TEST_CLAUSE_KWDS, 'r') as f:
        test_clause_kwds = json.load(f)

    ic(len(train_clause_kwds), len(dev_clause_kwds), len(test_clause_kwds))

    model_config = BartConfig(PRETRAINED_MODEL_NAME)

    ic(model_config)

    tokenizer = BartTokenizerFast.from_pretrained(PRETRAINED_MODEL_NAME)
    model = BartForConditionalGeneration.from_pretrained(PRETRAINED_MODEL_NAME)
    model.config.max_length = OUTPUT_MAX_LENGTH
    model.to(DEVICE)

    ic(tokenizer, model)

    train_ds = KwdToClauseDataset(train_clause_kwds, clause_kwd_limit=CLAUSE_KWD_LIMIT)
    dev_ds = KwdToClauseDataset(dev_clause_kwds, clause_kwd_limit=CLAUSE_KWD_LIMIT)
    test_ds = KwdToClauseDataset(test_clause_kwds, clause_kwd_limit=CLAUSE_KWD_LIMIT)

    ic(len(train_ds), len(dev_ds), len(test_ds))

    # from torch.utils.data import Subset

    # train_ds = Subset(train_ds, random.sample(range(len(train_ds)), 1000))
    # dev_ds = Subset(dev_ds, random.sample(range(len(train_ds)), 1000))
    # test_ds = Subset(test_ds, random.sample(range(len(train_ds)), 1000))
    # dev_ds = train_ds

    # ic(len(train_ds), len(dev_ds), len(test_ds))

    collate_fn = partial(collate_top_to_clause, tokenizer=tokenizer, 
        input_max_length=INPUT_MAX_LENGTH, output_max_length=OUTPUT_MAX_LENGTH)

    train_dl = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True, 
        num_workers=NUM_DATALOADER_PROCESSES, collate_fn=collate_fn)
    dev_dl = DataLoader(dev_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, 
        num_workers=NUM_DATALOADER_PROCESSES, collate_fn=collate_fn)
    test_dl = DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, 
        num_workers=NUM_DATALOADER_PROCESSES, collate_fn=collate_fn)
    train_dl, dev_dl, test_dl = accelerator.prepare(train_dl, dev_dl, test_dl)

    ic(len(train_dl), len(dev_dl), len(test_dl))

    optimizer = AdamW(model.parameters(), **OPTIMIZER_ARGS)

    num_train_update_steps = N_EPOCHS * len(train_dl) // ACCUMULATE_TRAIN_BATCHES
    SCHEDULER_ARGS = {
        'num_warmup_steps': num_train_update_steps // 4,
        'num_training_steps': num_train_update_steps,
    }
    scheduler = get_linear_schedule_with_warmup(optimizer, **SCHEDULER_ARGS)

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    ic(model, optimizer, scheduler)

    DATA_FILES = {
        'train_topic_clause_kwds_file': TRAIN_CLAUSE_KWDS,
        'dev_topic_clause_kwds_file': DEV_CLAUSE_KWDS,
        'test_topic_clause_kwds_file': TEST_CLAUSE_KWDS,
        'graph_file': 'None'
    }

    DATASET_INFO = {
        'train_dataset_size': len(train_ds),
        'dev_dataset_size': len(dev_ds),
        'test_dataset_size': len(test_ds),
        'max_keywords_per_clause': CLAUSE_KWD_LIMIT,
        'input_max_length': INPUT_MAX_LENGTH,
        'output_max_length': OUTPUT_MAX_LENGTH
    }

    MODEL_INFO = {
        'model_class': PRETRAINED_MODEL_NAME,
        'model_config': model_config
    }

    TRAIN_INFO = {
        'per_device_train_batch_size': TRAIN_BATCH_SIZE,
        'per_device_eval_batch_size': EVAL_BATCH_SIZE,
        'accumulate_train_batches': ACCUMULATE_TRAIN_BATCHES,
        'num_dataloder_processes': NUM_DATALOADER_PROCESSES,
        'total_epochs': N_EPOCHS,
        'per_device_train_steps': len(train_dl),
        'per_device_dev_steps': len(dev_dl),
        'per_device_test_steps': len(test_dl),
        'log_steps': LOG_STEPS,
        'validate_per_epoch': VALIDATE_PER_EPOCH,
        'validate_every_n_steps': int(len(train_dl) * VALIDATE_PER_EPOCH),
        'dev_bleu_every_k_random_steps': DEV_BLEU_EVERY_K_RANDOM_STEPS,
        'curr_epoch': 0,
        'curr_step': 0,
        'best_dev_bleu': float('-inf'),
        'avg_step_train_losses': list(),
        'avg_dev_losses': list(),
        'dev_bleus': list(),
        'avg_ms_per_batch': list(),
        'dev_durations': list(),
        'epoch_durations': list()
    }

    COMMENTS = ''

    SAVE_DICT = {
        'experiment_name': EXP_NAME,
        'model_info': MODEL_INFO,
        'model_state_dict': {},
        'optimizer_args': OPTIMIZER_ARGS,
        'scheduler_args': SCHEDULER_ARGS,
        'data_files': DATA_FILES,
        'dataset_info': DATASET_INFO,
        'train_info': TRAIN_INFO,
        'comments': COMMENTS
    }

    accelerator.print('State:', SAVE_DICT)

    if accelerator.is_local_main_process:
        logger = wandb.init(project=EXP_NAME, config=SAVE_DICT)
        # logger.watch(model)
        train(train_dl, dev_dl, tokenizer, model, optimizer, scheduler, accelerator, SAVE_DICT, logger)
    else:
        train(train_dl, dev_dl, tokenizer, model, optimizer, scheduler, accelerator, SAVE_DICT, None)

    wandb.finish()
