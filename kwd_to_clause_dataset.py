from copy import deepcopy

from torch.utils.data import Dataset

from utils import search_smallest_index_with_greater_value


class KwdToClauseDataset(Dataset):
    def __init__(self, topic_clause_kwds, clause_kwd_limit=50):
        self.topic_clause_kwds = topic_clause_kwds
        self.clause_kwd_limit = clause_kwd_limit
        
        self.prepare_data()
        self.build_cumulative_index()


    def __len__(self):
        return self.cum_size


    def __getitem__(self, idx):
        topic_indx = search_smallest_index_with_greater_value(self.topic_cum_sizes, idx)
        topic = list(self.topic_cum_index.keys())[topic_indx]
        clause_kwds = self.topic_clause_kwds[topic]
        
        if topic_indx > 0:
            prev_cum_size = self.topic_cum_sizes[topic_indx-1]
        else:
            prev_cum_size = -1
        
        clause, kwds = clause_kwds[idx - prev_cum_size - 1]

        return topic, kwds, clause


    def prepare_data(self):
        topic_clause_kwds = dict()
        for topic, clause_kwds in self.topic_clause_kwds.items():
            clause_kwds = [(clause, kwds[:self.clause_kwd_limit]) 
                for clause, kwds in clause_kwds]
            topic_clause_kwds[topic] = clause_kwds
        self.topic_clause_kwds = topic_clause_kwds

        return

    
    def build_cumulative_index(self):
        self.topic_cum_index = dict()

        self.cum_size = 0
        for topic in self.topic_clause_kwds:
            self.cum_size += len(self.topic_clause_kwds[topic])
            self.topic_cum_index[topic] = self.cum_size - 1
        self.topic_cum_sizes = list(self.topic_cum_index.values())

        return self.topic_cum_index


def collate_top_to_clause(batch, tokenizer, input_max_length, output_max_length):
    topic_strings = list()
    clauses = list()
    
    for topic, _, clause in batch:
        topic_strings.append(topic)
        clauses.append(clause)

    input_batch = tokenizer(topic_strings, return_tensors='pt', 
        add_special_tokens=True, return_attention_mask=True, 
        max_length=input_max_length, padding='longest', truncation=True)
    output_batch = tokenizer(clauses, return_tensors='pt', 
        add_special_tokens=True, return_attention_mask=True, 
        max_length=output_max_length, padding='longest', truncation=True)
    output_batch['input_ids'][output_batch['attention_mask'] == 0] = -100

    return_batch = input_batch
    return_batch['labels'] = output_batch['input_ids']

    return return_batch


def collate_kwd_to_clause(batch, tokenizer, input_max_length, output_max_length):
    sep_token = tokenizer.special_tokens_map['sep_token']
    topic_kwd_strings = list()
    clauses = list()
    
    for topic, kwds, clause in batch:
        topic_kwd_strings.append(topic + sep_token + ' '.join(kwds))
        clauses.append(clause)

    input_batch = tokenizer(topic_kwd_strings, return_tensors='pt', 
        add_special_tokens=True, return_attention_mask=True, 
        max_length=input_max_length, padding='longest', truncation=True)
    output_batch = tokenizer(clauses, return_tensors='pt', 
        add_special_tokens=True, return_attention_mask=True, 
        max_length=output_max_length, padding='longest', truncation=True)
    output_batch['input_ids'][output_batch['attention_mask'] == 0] = -100

    return_batch = input_batch
    return_batch['labels'] = output_batch['input_ids']

    return return_batch


def collate_kwd_to_clause_for_causal_lm_loss(batch, tokenizer, topic_sep_token, seq_max_length):
    return_batch = tokenizer(
        [t + topic_sep_token + ' '.join(k) + topic_sep_token + c + tokenizer.eos_token for t, k, c in batch],
        add_special_tokens=True, return_attention_mask=True, return_tensors='pt',
        max_length=seq_max_length, padding='longest', truncation=True
    )
    
    labels = deepcopy(return_batch['input_ids'])
    labels[return_batch['attention_mask'] == 0] = -100
    return_batch['labels'] = labels

    return return_batch


def collate_kwd_to_clause_for_causal_lm_generation(batch, tokenizer, topic_sep_token, 
    prompt_max_length, seq_max_length):

    prompts_batch = tokenizer(
        [t + topic_sep_token + ' '.join(k) + topic_sep_token for t, k, _ in batch],
        add_special_tokens=True, return_attention_mask=True, return_tensors='pt',
        max_length=prompt_max_length, padding='longest', truncation=True
    )
    clauses_batch = tokenizer(
        [c + tokenizer.eos_token for _, _, c in batch],
        add_special_tokens=True, return_attention_mask=True, return_tensors='pt',
        max_length=seq_max_length, padding='longest', truncation=True
    )['input_ids']

    return_batch = {
        'input_ids': prompts_batch['input_ids'],
        'attention_mask': prompts_batch['attention_mask'],
        'labels': clauses_batch
    }

    return return_batch
