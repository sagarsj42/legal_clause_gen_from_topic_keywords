import json
import pickle

import numpy as np
import scipy.sparse as sp


def create_adj_mat_from_nostep_kwds(topic_kwds, vocab, maxlim=10000):
    vocab_size = len(vocab)
    eos_indx = vocab.lookup_indices(['<EOS>'])[0]
    adj_mat = np.zeros((vocab_size, vocab_size))

    for topic, kwd_lists in topic_kwds.items():
        topic_indx = vocab.lookup_indices([topic])[0]
        for clause_kwd_list in kwd_lists:
            p_kwd_indx = None
            for i, kwd in enumerate(clause_kwd_list[:maxlim]):
                kwd_indx = vocab.lookup_indices([kwd])[0]
                adj_mat[topic_indx, kwd_indx] += 1.0/(i+1)
                if i != 0:
                    adj_mat[p_kwd_indx, kwd_indx] += 1.0/(i+1)
                p_kwd_indx = kwd_indx
            adj_mat[p_kwd_indx, eos_indx] += 1.0/(i+1)
    adj_mat = sp.csr_matrix(adj_mat)

    return adj_mat


def create_adj_mat_from_nostep_kwds_with_topic_edge_limit(topic_kwds, vocab, topic_edge_limit=10, 
    clause_kwd_limit=50):
    
    vocab_size = len(vocab)
    eos_indx = vocab.lookup_indices(['<EOS>'])[0]
    topic_sizes = {t: len(c) for t, c in topic_kwds.items()}
    adj_mat = np.zeros((vocab_size, vocab_size))

    for topic, clause_kwd_lists in topic_kwds.items():
        topic_indx = vocab.lookup_indices([topic])[0]
        for _, kwd_list in clause_kwd_lists:
            p_kwd_indx = None
            for i, kwd in enumerate(kwd_list[:clause_kwd_limit]):
                kwd_indx = vocab.lookup_indices([kwd])[0]
                normalizer = (i+1) * topic_sizes[topic]
                if i < topic_edge_limit:
                    adj_mat[topic_indx, kwd_indx] += 1.0/normalizer
                if i != 0:
                    adj_mat[p_kwd_indx, kwd_indx] += 1.0/normalizer
                p_kwd_indx = kwd_indx
            adj_mat[p_kwd_indx, eos_indx] += 1.0/(i+1)
    adj_mat = sp.csr_matrix(adj_mat)

    return adj_mat


if __name__ == '__main__':
    with open('topic-nostep-kwds.lemstop.train.json', 'r') as f:
        train_nostep_topic_kwds = json.load(f)
    
    with open('topic-keywords-vocab.pkl', 'rb') as f:
        all_vocab = pickle.load(f)

    adj_mat_nostep = create_adj_mat_from_nostep_kwds(train_nostep_topic_kwds, all_vocab)
    sp.save_npz('adj-mat.nostep.npz', adj_mat_nostep, compressed=True)
    print('Adjacency matrix constructed using no-step keywords.')

    maxlim = 50
    adj_mat_nostep_50 = create_adj_mat_from_nostep_kwds(train_nostep_topic_kwds, all_vocab, maxlim=maxlim)
    sp.save_npz(f'adj-mat.nostep.{maxlim}.npz', adj_mat_nostep, compressed=True)
    print(f'Adjacency matrix constructed using no-step keywords and maxlim {maxlim}.')
