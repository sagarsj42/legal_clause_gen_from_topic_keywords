import os
import json
from functools import partial

import torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

import numpy as np
from nltk.translate import bleu_score
from rouge_score import rouge_scorer
from icecream import ic

from kwd_to_clause_dataset import KwdToClauseDataset, \
    collate_kwd_to_clause_for_causal_lm_generation


def collate_prompt_to_clause_for_causal_lm_generation(batch, tokenizer, topic_sep_token, 
    prompt_max_length, seq_max_length):

    prompts_batch = tokenizer(
        [t + topic_sep_token + ' '.join(c.split()[:prompt_max_length]) for t, _, c in batch],
        add_special_tokens=True, return_attention_mask=True, return_tensors='pt',
        max_length=2*prompt_max_length, padding='longest', truncation=True
    )
    clauses_batch = tokenizer(
        [' '.join(c.split()[prompt_max_length:]) for _, _, c in batch],
        add_special_tokens=True, return_attention_mask=True, return_tensors='pt',
        max_length=seq_max_length, padding='longest', truncation=True
    )['input_ids']

    return_batch = {
        'input_ids': prompts_batch['input_ids'],
        'attention_mask': prompts_batch['attention_mask'],
        'labels': clauses_batch
    }

    return return_batch


def add_rouge_scores_to_dict(rouge_strs, rouge_score, res_dict):
    for rouge_str in rouge_strs:
        res_dict[rouge_str] = {
            'p': rouge_score[rouge_str].precision,
            'r': rouge_score[rouge_str].recall,
            'f': rouge_score[rouge_str].fmeasure
        }

    return res_dict


def calculate_aggregate_rouge(rouge_strs, all_res):
    rouge_agg = dict()
    for rouge_str in rouge_strs:
        measure_list_dict = {
            'p': list(),
            'r': list(),
            'f': list()
        }
        
        for res in all_res:
            for measure in res[rouge_str]:
                measure_list_dict[measure].append(res[rouge_str][measure])
        
        p = np.array(measure_list_dict['p']).mean()
        r = np.array(measure_list_dict['r']).mean()
        f = np.array(measure_list_dict['f']).mean()
        rouge_agg[rouge_str] = {'p': p, 'r': r, 'f': f}

    return rouge_agg


def calculate_average_metric_val(metric, all_res):
    agg = np.array([res[metric] for res in all_res]).mean()

    return agg


def calculate_corpus_bleu(all_res):
    all_refs = [[res['ref']] for res in all_res]
    all_outs = [res['out'] for res in all_res]
    corpus_bleu = bleu_score.corpus_bleu(all_refs, all_outs)

    return corpus_bleu


def evaluate_generation(tops, kwds, refs, outs):
    rouge_score_strs = ['rouge1', 'rouge2', 'rougeL']
    scorer = rouge_scorer.RougeScorer(rouge_score_strs, use_stemmer=True)
    all_res = list()

    for t, k, r, o in zip(tops, kwds, refs, outs):
        res = dict()
        res['topic'] = t
        res['kwds'] = k
        res['ref'] = r
        res['out'] = o
        
        r_t = r.split()
        o_t = o.split()
        res['sent_bleu'] = bleu_score.sentence_bleu([r_t], o_t)
        rouge_scores = scorer.score(r, o)
        add_rouge_scores_to_dict(rouge_score_strs, rouge_scores, res)

        all_res.append(res)
    
    metrics = {
        'sent_bleu': calculate_average_metric_val('sent_bleu', all_res),
        'corpus_bleu': calculate_corpus_bleu(all_res)
    }
    rouge_metrics = calculate_aggregate_rouge(rouge_score_strs, all_res)
    for rouge_str in rouge_score_strs:
        metrics[rouge_str] = rouge_metrics[rouge_str]

    return all_res, metrics


def get_output_clauses(dataset, batch_size, collate_fn, tokenizer, model, device='cpu'):
    
    batch_lim_list = list(range(0, len(dataset), batch_size))
    tops = list()
    kwds = list()
    refs = list()
    outs = list()

    for i in range(len(batch_lim_list)-1):
        start = batch_lim_list[i]
        end = batch_lim_list[i+1]
        batch = [dataset[i] for i in range(start, end)]

        if i % 100 == 0:
            ic(start, end)
        
        collated_batch = collate_fn(batch)
        collated_batch = {k: v.to(device) for k, v in collated_batch.items()}

        for t, k, r in batch:
            tops.append(t)
            kwds.append(k)
            refs.append(r)
        
        for b in range(batch_size):
            inp = collated_batch['input_ids'][b, :]
            inp = inp[collated_batch['attention_mask'][b, :] == 1].unsqueeze(0)
            inp_len = len(tokenizer.decode(inp[0, :], skip_special_tokens=True))
            
            out = tokenizer.decode(model.generate(inp, 
                pad_token_id=tokenizer.eos_token_id)[0, :], skip_special_tokens=True)
            out = out[inp_len:]
            outs.append(out)

    return tops, kwds, refs, outs


def get_output_clauses_with_evaluation(dataset, batch_size, collate_fn, 
    tokenizer, model, device='cpu'):

    tops, kwds, refs, outs = get_output_clauses(dataset, batch_size, collate_fn, 
        tokenizer, model, device=device)
    
    all_res, agg_metrics = evaluate_generation(tops, kwds, refs, outs)

    output = {
        'individual': all_res,
        'aggregate': agg_metrics
    }

    return output


if __name__ == '__main__':
    os.chdir('/scratch/sagarsj42')
    os.environ['TORCH_HOME'] = '/scratch/sagarsj42/torch-cache'
    os.environ['TRANSFORMERS_CACHE'] = '/scratch/sagarsj42/transformers'

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TRAIN_CLAUSE_KWDS = 'topic-nostep-clause-kwds.c100.k200.lemstop.train.json'
    DEV_CLAUSE_KWDS = 'topic-nostep-clause-kwds.c100.k200.lemstop.dev.json'
    TEST_CLAUSE_KWDS = 'topic-nostep-clause-kwds.c100.k200.lemstop.test.json'

    PRETRAINED_MODEL_NAME = 'gpt2'
    EXP_NAME = 'kwd-to-clause-gpt2'
    MODEL_PATH = os.path.join(EXP_NAME, 'best.pth')
    TOPIC_SEP_TOKEN = '</s>'

    CLAUSE_KWD_LIMIT = 10
    PROMPT_MAX_LENGTH = 50
    SEQUENCE_MAX_LENGTH = 800
    EVAL_BATCH_SIZE = 8
    
    tokenizer = GPT2TokenizerFast.from_pretrained(PRETRAINED_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    ic(tokenizer)

    model = GPT2LMHeadModel.from_pretrained(PRETRAINED_MODEL_NAME)
    model.config.max_length = SEQUENCE_MAX_LENGTH
    save_dict = torch.load(MODEL_PATH)
    model.load_state_dict(save_dict['model_state_dict'])
    model.to(DEVICE)

    ic(model)

    with open(TEST_CLAUSE_KWDS, 'r') as f:
        test_clause_kwds = json.load(f)

    ic(len(test_clause_kwds))

    test_ds = KwdToClauseDataset(test_clause_kwds, clause_kwd_limit=CLAUSE_KWD_LIMIT)

    # from torch.utils.data import Subset
    # test_ds = Subset(test_ds, range(100))

    ic(len(test_ds))

    collate_fn_gen = partial(collate_kwd_to_clause_for_causal_lm_generation, tokenizer=tokenizer, 
        topic_sep_token=TOPIC_SEP_TOKEN, prompt_max_length=PROMPT_MAX_LENGTH, 
        seq_max_length=SEQUENCE_MAX_LENGTH)

    ic(collate_fn_gen)

    eval_op = get_output_clauses_with_evaluation(test_ds, batch_size=EVAL_BATCH_SIZE, 
        collate_fn=collate_fn_gen, tokenizer=tokenizer, model=model, device=DEVICE)

    ic(eval_op.keys())
    ic(eval_op['aggregate'])

    with open(EXP_NAME + '.results.json', 'w') as f:
        json.dump(eval_op, f)
