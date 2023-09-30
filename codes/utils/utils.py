import re

import nltk
import torch
import numpy as np


def clean_clause(clause, lowercase=True, normalize_all_caps=True):
    if not isinstance(clause, str):
        raise ValueError('Input clause is not a string.')
    
    cleaned = re.sub(r'[^A-Za-z0-9\s/()_\-&.,;"\':$%]', '', clause)
    cleaned = re.sub(r'__+', r'__', cleaned)
    cleaned = re.sub(r'\s\s+', ' ', cleaned)
    
    if lowercase:
        cleaned = cleaned.lower()
        
    if cleaned.isupper() and normalize_all_caps:
        cleaned = '. '.join(sent.capitalize() for sent in cleaned.split('. '))
        
    cleaned = cleaned.strip()
    
    return cleaned


def clean_topic(topic, lowercase=True):
    if not isinstance(topic, str):
        raise ValueError('Input topic is not a string.')
    
    if lowercase:
        topic = topic.lower()

    topic = re.sub(r'[^a-z0-9]', ' ', topic)
    topic = re.sub(r'\s\s+', ' ', topic)
    topic = topic.strip()

    return topic


def search_smallest_index_with_greater_value(array, search_val):
    s = 0
    e = len(array) - 1
    m = (s + e) // 2
    
    while True:
        if s > e:
            return -1
        elif search_val == array[m] or s == e:
            return m
        elif search_val < array[m]:
            e = m
        elif search_val > array[m]:
            s = m + 1
        m = (s + e) // 2


def load_embeds_with_pretrained(vocab, pretrained_embeds, embed_size=300):
    embeds = list()
    oov = list()

    for i in range(len(vocab)):
        word = vocab.lookup_token(i)
        try:
            embeds.append(pretrained_embeds[word])
        except:
            oov.append(word)
            embeds.append(np.zeros(embed_size))
    embeds = torch.tensor(np.array(embeds), dtype=torch.double)

    return embeds, oov


def topic_to_word_indx(topic, topics_vocab, lemmatizer=None, stopwords=None):
    if not lemmatizer:
        lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    if not stopwords:
        stopwords = set(nltk.corpus.stopwords.words('english'))
    
    topic_words = list()
    for topic_word in topic.split():
        topic_word = lemmatizer.lemmatize(re.sub(r'[^\w]', '', topic_word))
        if topic_word not in stopwords:
            topic_words.append(topic_word)
    topic_word_indx = topics_vocab.lookup_indices(topic_words)

    return topic_word_indx
