import time
import json
import pickle
import random
from functools import partial

import torch
import nltk
import numpy as np
import networkx as nx
import scipy.sparse as sp

from clsgen.top_kwd_coherence_model import TopicKeywordCoherenceScorer_2
from clsgen.kwd_to_kwd_coh_model import KeywordToKeywordCoherenceScorer_3
from clsgen.graph_plan_ranking import plan_rank_simplesum, aggregate_planrank_stats
from clsgen.utils import topic_to_word_indx


def obtain_plan_ranks_from_clause_kwds(clause_kwds, plan_ranker, max_clauses):
    start = time.time()
    topic_ref_plan_ranks = dict()

    for topic, ref_plans in clause_kwds.items():
        # if len(ref_plans) < 5000 or len(ref_plans) > 500000:
        #     continue
        n_clauses = len(clause_kwds[topic])
        n_samples = n_clauses
        n_samples = min(n_samples, max_clauses)
        select_plan_indices = random.sample(range(n_clauses), n_samples)
        ref_plan_ranks = list()

        for spi in select_plan_indices:
            ref_plan = ref_plans[spi][1]
            if len(ref_plan) == 0:
                continue

            plan_ranks, n_neighbors = plan_ranker(topic, ref_plan=ref_plan, graph=graph, 
                all_vocab=all_vocab, n_steps=N_WALK_STEPS)

            ref_plan_ranks.append((spi, ref_plan, plan_ranks, n_neighbors))
        topic_ref_plan_ranks[topic] = ref_plan_ranks
    plan_time = time.time() - start
    n_plans = sum([len(c) for c in topic_ref_plan_ranks.values()])

    print('Plan time:', plan_time)
    print('# plans:', n_plans)

    return topic_ref_plan_ranks


def print_planrank_stats(topic_ref_plan_ranks, split_name):
    planrank_stats = aggregate_planrank_stats(topic_ref_plan_ranks)

    print(f'Split: {split_name}')
    print(planrank_stats['aggregated'])

    n_instances = [len(v) for v in planrank_stats['collated']['stagewise_n_neigh'].values()]
    rank_means = planrank_stats['stagewise']['rank']['mean']
    rank_medians = planrank_stats['stagewise']['rank']['median']
    n_neigh_means = planrank_stats['stagewise']['n_neigh']['mean']
    n_neigh_medians = planrank_stats['stagewise']['n_neigh']['median']

    print('Stage-wise values of # instances at that stage, mean rank, median rank, mean # neighbors, median # neighbors')
    for nins, rmean, emed, nmean, nmed in zip(n_instances, rank_means, rank_medians, n_neigh_means, n_neigh_medians):
        print(f'{nins}\t{rmean}\t{emed}\t{nmean}\t{nmed}')
    
    return


if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    MIN_CLAUSES = 100
    MAX_TOPIC_KWDS = 200
    TOPIC_EDGE_LIMIT = 10
    CLAUSE_KWD_LIMIT = 10
    N_WALK_STEPS = 10

    TRAIN_CLAUSE_KWDS = f'topic-nostep-clause-kwds.c{MIN_CLAUSES}.k{MAX_TOPIC_KWDS}.lemstop.train.json'
    DEV_CLAUSE_KWDS = f'topic-nostep-clause-kwds.c{MIN_CLAUSES}.k{MAX_TOPIC_KWDS}.lemstop.dev.json'
    TEST_CLAUSE_KWDS = f'topic-nostep-clause-kwds.c{MIN_CLAUSES}.k{MAX_TOPIC_KWDS}.lemstop.test.json'
    GRAPH_FILE = f'adj-mat.nostep.t{TOPIC_EDGE_LIMIT}-k{CLAUSE_KWD_LIMIT}.npz'
    ALL_VOCAB_FILE = 'topic-keywords-vocab.pkl'
    TOPIC_VOCAB_FILE = 'topic-vocab.pkl'
    KWDS_VOCAB_FILE = 'keywords-vocab.pkl'
    TOP_KWD_COH_MODEL_PATH = 'topic-kwd-mrr-3.2/best.pth'
    KWD_KWD_COH_MODEL_PATH = 'kwd-kwd-mrr-2.3/best.pth'

    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    stopwords = set(nltk.corpus.stopwords.words('english'))

    with open(TRAIN_CLAUSE_KWDS, 'r') as f:
        train_clause_kwds = json.load(f)
    with open(DEV_CLAUSE_KWDS, 'r') as f:
        dev_clause_kwds = json.load(f)
    with open(TEST_CLAUSE_KWDS, 'r') as f:
        test_clause_kwds = json.load(f)
    
    adj_mat = sp.load_npz(GRAPH_FILE)
    graph = nx.from_scipy_sparse_array(adj_mat, create_using=nx.DiGraph)

    with open(ALL_VOCAB_FILE, 'rb') as f:
        all_vocab = pickle.load(f)
    with open(TOPIC_VOCAB_FILE, 'rb') as f:
        topic_vocab = pickle.load(f)
    with open(KWDS_VOCAB_FILE, 'rb') as f:
        kwd_vocab = pickle.load(f)

    ALL_VOCAB_SIZE = len(all_vocab)
    TOPIC_VOCAB_SIZE = len(topic_vocab)
    KWD_VOCAB_SIZE = len(kwd_vocab)

    train_topic_ref_plan_ranks = obtain_plan_ranks_from_clause_kwds(train_clause_kwds, 
        plan_ranker=plan_rank_simplesum, max_clauses=10)
    
    print_planrank_stats(train_topic_ref_plan_ranks, split_name='train')
    
    dev_topic_ref_plan_ranks = obtain_plan_ranks_from_clause_kwds(dev_clause_kwds, 
        plan_ranker=plan_rank_simplesum, max_clauses=10)
    
    print_planrank_stats(dev_topic_ref_plan_ranks, split_name='dev')
    
    test_topic_ref_plan_ranks = obtain_plan_ranks_from_clause_kwds(test_clause_kwds, 
        plan_ranker=plan_rank_simplesum, max_clauses=10)
    
    print_planrank_stats(test_topic_ref_plan_ranks, split_name='test')
