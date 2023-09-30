import re
import time
import random
import json
import pickle
from collections import defaultdict, Counter

import yake
import nltk
import pandas as pd
import seaborn as sns
from torchtext.vocab import vocab, build_vocab_from_iterator
from matplotlib import pyplot as plt

from utils import clean_clause, clean_topic


def add_topic_clauses_from_df(df, topic_clauses):
    for i in range(df.shape[0]):
        topics = df.iloc[i]['label']
        if len(topics) > 1:
            continue
        clause = clean_clause(df.iloc[i]['provision'])
        topic = clean_topic(topics[0])
        topic_clauses[topic].append(clause)

    return topic_clauses



def get_topic_keywords_dict(topic_clauses):
    topic_kwds = dict()
    kw_extractor = yake.KeywordExtractor(lan='en', n=1, dedupLim=1.0, top=10000)

    for topic in topic_clauses:
        concat_clauses = '\n'.join(topic_clauses[topic])
        yake_keywords = kw_extractor.extract_keywords(concat_clauses)
        topic_kwds[topic] = yake_keywords

    return topic_kwds


def filter_topic_kwds_by_lemstop(topic_kwds):
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    stopwords = set(nltk.corpus.stopwords.words('english'))
    topic_lemstop_kwds = dict()

    for topic, kwds in topic_kwds.items():
        lemstop_kwds = list()
        kwd_set = set()
        for s, k in kwds:
            k = re.sub(r'[^\w]', '', k)
            k = lemmatizer.lemmatize(k)
            if k not in stopwords:
                if k not in kwd_set:
                    kwd_set.add(k)
                    lemstop_kwds.append((s, k))
        topic_lemstop_kwds[topic] = lemstop_kwds

    return topic_lemstop_kwds


def get_nostep_kwds_from_topic_kwds(topic_kwds, topic_clauses, maxlim=1000):
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    stopwords = set(nltk.corpus.stopwords.words('english'))
    topic_stepped_kwds = dict()
    
    for topic, clauses in topic_clauses.items():
        kwds = [k for _, k in topic_kwds[topic]][:maxlim]
        nostep_kwds = list()

        for clause in clauses:
            clause_wordset = set([lemmatizer.lemmatize(re.sub(r'[^\w]', '', word)) 
                for word in clause.split() if word not in stopwords])
            nostep_kwds.append([kwd for kwd in kwds if kwd in clause_wordset])
        topic_stepped_kwds[topic] = nostep_kwds

    return topic_stepped_kwds


def yield_tokens(wordlist):
    yield wordlist


if __name__ == '__main__':
    random.seed(42)
    start = time.time()
    filename = 'LEDGAR_2016-2019_clean.jsonl'
    ledgar = pd.read_json(filename, lines=True)

    print('Time to load LEDGAR:', time.time() - start)
    
    contracts = list(set(ledgar['source']))
    n_contr = len(contracts)

    print('# contracts in LEDGAR:', len(contracts))

    train_ratio = 0.85
    dev_ratio = 0.05
    test_ratio = 0.10

    train_contr_indices = random.sample(list(range(n_contr)), int(n_contr*train_ratio))
    dev_contr_indices = random.sample(list(set(range(n_contr)) - set(train_contr_indices)), 
        int(n_contr*dev_ratio))
    test_contr_indices = random.sample(list(set(range(n_contr)) - set(dev_contr_indices) \
        - set(train_contr_indices)), int(n_contr*test_ratio))
    
    train_contrs = set([contracts[i] for i in train_contr_indices])
    dev_contrs = set([contracts[i] for i in dev_contr_indices])
    test_contrs = set([contracts[i] for i in test_contr_indices])
    
    print(f'train contracts: {len(train_contrs)}, dev contracts: {len(dev_contrs)}, test_contracts: {len(test_contrs)}')

    train_topic_clauses = defaultdict(list)
    dev_topic_clauses = defaultdict(list)
    test_topic_clauses = defaultdict(list)
    all_topic_clauses = defaultdict(list)

    for group in ledgar.groupby('source'):
        contr_name = group[0]
        contr_df = group[1]
        if contr_name in train_contrs:
            add_topic_clauses_from_df(contr_df, train_topic_clauses)
        elif contr_name in dev_contrs:
            add_topic_clauses_from_df(contr_df, dev_topic_clauses)
        elif contr_name in test_contrs:
            add_topic_clauses_from_df(contr_df, test_topic_clauses)
        add_topic_clauses_from_df(contr_df, all_topic_clauses)
    
    train_topic_sizes = {t: s for t, s in zip(train_topic_clauses.keys(), 
        map(lambda l : len(l), train_topic_clauses.values()))}
    dev_topic_sizes = {t: s for t, s in zip(dev_topic_clauses.keys(), 
        map(lambda l : len(l), dev_topic_clauses.values()))}
    test_topic_sizes = {t: s for t, s in zip(test_topic_clauses.keys(), 
        map(lambda l : len(l), test_topic_clauses.values()))}

    print(f'# train topics: {len(train_topic_sizes)}, # dev topics: {len(dev_topic_sizes)}, \
        # test topics: {len(test_topic_sizes)}')
    print('Max # clauses per topic:')
    print(f'train: {max(list(train_topic_sizes.values()))}, dev: {max(list(dev_topic_sizes.values()))}, \
        test: {max(list(test_topic_sizes.values()))}')
    
    fig, ax = plt.subplots(nrows=5, ncols=3, figsize=(24, 24))
    sns.histplot(list(train_topic_sizes.values()), binrange=(100, 11000), ax=ax[0][0])
    ax[0][0].set_title('Distribution of # clauses per topic: train\nMinimum # clauses per topic: 1')
    ax[0][0].set_xlabel('# clause topics')
    sns.histplot(list(train_topic_sizes.values()), binrange=(100, 11000), ax=ax[1][0])
    ax[1][0].set_title('Minimum # clauses per topic: 100')
    ax[1][0].set_xlabel('# clause topics')
    sns.histplot(list(train_topic_sizes.values()), binrange=(500, 11000), ax=ax[2][0])
    ax[2][0].set_title('Minimum # clauses per topic: 500')
    ax[2][0].set_xlabel('# clause topics')
    sns.histplot(list(train_topic_sizes.values()), binrange=(800, 11000), ax=ax[3][0])
    ax[3][0].set_title('Minimum # clauses per topic: 800')
    ax[3][0].set_xlabel('# clause topics')
    sns.histplot(list(train_topic_sizes.values()), binrange=(1000, 11000), ax=ax[4][0])
    ax[4][0].set_title('Minimum # clauses per topic: 1000')
    ax[4][0].set_xlabel('# clause topics')

    sns.histplot(list(dev_topic_sizes.values()), binrange=(1, 700), ax=ax[0][1], color='y')
    ax[0][1].set_title('Distribution of # clauses per topic: dev\nMinimum # clauses per topic: 1')
    ax[0][1].set_xlabel('# clause topics')
    sns.histplot(list(dev_topic_sizes.values()), binrange=(5, 700), ax=ax[1][1], color='y')
    ax[1][1].set_title('Minimum # clauses per topic: 5')
    ax[1][1].set_xlabel('# clause topics')
    sns.histplot(list(dev_topic_sizes.values()), binrange=(25, 700), ax=ax[2][1], color='y')
    ax[2][1].set_title('Minimum # clauses per topic: 25')
    ax[2][1].set_xlabel('# clause topics')
    sns.histplot(list(dev_topic_sizes.values()), binrange=(40, 700), ax=ax[3][1], color='y')
    ax[3][1].set_title('Minimum # clauses per topic: 40')
    ax[3][1].set_xlabel('# clause topics')
    sns.histplot(list(dev_topic_sizes.values()), binrange=(50, 700), ax=ax[4][1], color='y')
    ax[4][1].set_title('Minimum # clauses per topic: 50')
    ax[4][1].set_xlabel('# clause topics')

    sns.histplot(list(test_topic_sizes.values()), binrange=(1, 1300), ax=ax[0][2], color='r')
    ax[0][2].set_title('Distribution of # clauses per topic: test\nMinimum # clauses per topic: 1')
    ax[0][2].set_xlabel('# clause topics')
    sns.histplot(list(test_topic_sizes.values()), binrange=(10, 1300), ax=ax[1][2], color='r')
    ax[1][2].set_title('Minimum # clauses per topic: 10')
    ax[1][2].set_xlabel('# clause topics')
    sns.histplot(list(test_topic_sizes.values()), binrange=(50, 1300), ax=ax[2][2], color='r')
    ax[2][2].set_title('Minimum # clauses per topic: 50')
    ax[2][2].set_xlabel('# clause topics')
    sns.histplot(list(test_topic_sizes.values()), binrange=(80, 1300), ax=ax[3][2], color='r')
    ax[3][2].set_title('Minimum # clauses per topic: 80')
    ax[3][2].set_xlabel('# clause topics')
    sns.histplot(list(test_topic_sizes.values()), binrange=(100, 1300), ax=ax[4][2], color='r')
    ax[4][2].set_title('Minimum # clauses per topic: 100')
    ax[4][2].set_xlabel('# clause topics')

    plt.tight_layout()
    plt.savefig('histograms-no-clauses-per-topic-all-splits.png', facecolor='w')
    plt.show()

    train_topic_kwds = get_topic_keywords_dict(train_topic_clauses)
    eval_topic_kwds = get_topic_keywords_dict(all_topic_clauses)

    with open('topic-kwds.train.json', 'w') as f:
        json.dump(train_topic_kwds, f)
    with open('topic-kwds.eval.json', 'w') as f:
        json.dump(eval_topic_kwds, f)
    
    train_topic_lemstop_kwds = filter_topic_kwds_by_lemstop(train_topic_kwds)
    eval_topic_lemstop_kwds = filter_topic_kwds_by_lemstop(eval_topic_kwds)

    with open('topic-kwds.lemstop.train.json', 'w') as f:
        json.dump(train_topic_lemstop_kwds, f)
    with open('topic-kwds.lemstop.eval.json', 'w') as f:
        json.dump(eval_topic_lemstop_kwds, f)
    
    train_nostep_kwds = get_nostep_kwds_from_topic_kwds(train_topic_lemstop_kwds, train_topic_clauses)
    dev_nostep_kwds = get_nostep_kwds_from_topic_kwds(eval_topic_lemstop_kwds, dev_topic_clauses)
    test_nostep_kwds = get_nostep_kwds_from_topic_kwds(eval_topic_lemstop_kwds, test_topic_clauses)

    with open('topic-nostep-kwds.lemstop.train.json', 'w') as f:
        json.dump(train_nostep_kwds, f)
    with open('topic-nostep-kwds.lemstop.dev.json', 'w') as f:
        json.dump(dev_nostep_kwds, f)
    with open('topic-nostep-kwds.lemstop.test.json', 'w') as f:
        json.dump(test_nostep_kwds, f)
    
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    stopwords = set(nltk.corpus.stopwords.words('english'))
    unk_token = '<UNK>'
    pad_token = '<PAD>'
    bos_token = '<BOS>'
    eos_token = '<EOS>'
    special_tokens = [unk_token, pad_token, bos_token, eos_token]

    topic_vocab = Counter()

    for topic in train_nostep_kwds:
        topic_words = topic.split()
        for topic_word in topic_words:
            topic_word = lemmatizer.lemmatize(re.sub(r'[^\w]', '', topic_word))
            if topic_word not in stopwords:
                topic_vocab.update([topic_word])

    topic_vocab = sorted(topic_vocab.items(), key=lambda x:x[1], reverse=True)
    topic_vocab = [t for t, _ in topic_vocab]
    topic_vocab = build_vocab_from_iterator(yield_tokens(topic_vocab), 
        specials=special_tokens, special_first=True)
    topic_vocab.set_default_index(0)

    with open('topic-vocab.pkl', 'wb') as f:
        pickle.dump(topic_vocab, f)
    
    keywords_vocab = Counter()

    for kwd_lists in train_nostep_kwds.values():
        for kwd_list in kwd_lists:
            for kwd in kwd_list:
                kwd = lemmatizer.lemmatize(re.sub(r'[^\w]', '', kwd))
                if kwd not in stopwords:
                    keywords_vocab.update([kwd])

    keywords_vocab = sorted(keywords_vocab.items(), key=lambda x:x[1], reverse=True)
    keywords_vocab = [k for k, _ in keywords_vocab]
    keywords_vocab = build_vocab_from_iterator(yield_tokens(keywords_vocab), 
        specials=special_tokens, special_first=True)
    keywords_vocab.set_default_index(0)

    with open('keywords-vocab.pkl', 'wb') as f:
        pickle.dump(keywords_vocab, f)
    
    keyword_set = set(keywords_vocab.get_itos()) - set(special_tokens)
    topic_set = set(train_topic_clauses.keys())
    all_words_set = list(topic_set.union(keyword_set))
    all_words_set.sort()

    all_vocab = build_vocab_from_iterator(yield_tokens(all_words_set), 
        specials=special_tokens, special_first=True)
    all_vocab.set_default_index(0)

    with open('topic-keywords-vocab.pkl', 'wb') as f:
        pickle.dump(all_vocab, f)
