import argparse
import csv
import os
import re
from itertools import chain

import numpy as np
import pandas as pd
from mittens import Mittens
from sklearn.feature_extraction.text import CountVectorizer

from word_tokenize import word_tokenize


def get_sents(data_dir, db_info):
    train_path = os.path.join(data_dir, 'train.csv')
    train = pd.read_csv(train_path, usecols=['source', 'labels'])
    train = pd.DataFrame(train)

    test_path = os.path.join(data_dir, 'test.csv')
    test = pd.read_csv(test_path, usecols=['source', 'labels'])
    test = pd.DataFrame(test)

    dev_path = os.path.join(data_dir, 'dev.csv')
    dev = pd.read_csv(dev_path, usecols=['source', 'labels'])
    dev = pd.DataFrame(dev)

    db = pd.read_csv(db_info)
    db = pd.DataFrame(db)

    sents = []
    for sent in chain(train['source'], train['labels'], test['source'], test['labels'],
                      dev['source'], dev['labels'], db['table'], db['column'], db['value']):
        if isinstance(sent, str):
            sents.append(sent.lower())

    return sents


def glove2dict(glove_path):
    with open(glove_path, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
        embed = {line[0]: np.array(list(map(float, line[1:]))) for line in reader}

    return embed


def get_vocab(sents):
    vocab = list(set(token for sent in sents for token in word_tokenize(sent)))

    return vocab


def get_cooccurrence(sents, vocab):
    count_model = CountVectorizer(ngram_range=(1, 1),
                                  vocabulary=vocab,
                                  tokenizer=word_tokenize,
                                  stop_words='english')
    X = count_model.fit_transform(sents)
    Xc = (X.T * X)
    Xc.setdiag(0)

    return Xc.toarray()


def get_embeddings(sents, pre_glove, n, max_iter):
    vocab = get_vocab(sents)
    cooccurrence = get_cooccurrence(sents, vocab)
    mittens_model = Mittens(n=n, max_iter=max_iter)
    embeddings = mittens_model.fit(cooccurrence, vocab=vocab, initial_embedding_dict=pre_glove)

    return vocab, embeddings


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='fine_tuning.py')
    parser.add_argument('-data_dir', required=False, default='./dataset/dataset_final/')
    parser.add_argument('-db_info', required=False, default='./dataset/database_information.csv')
    parser.add_argument('-glove_path', required=False, default='./dataset/glove.6B.100d.txt')
    parser.add_argument('-tuned_glove_path',
                        required=False,
                        default='./dataset/tuned_glove.6B.100d.txt')
    parser.add_argument('-n', required=False, default=100)
    parser.add_argument('-max_iter', required=False, default=10000)
    opt = parser.parse_args()

    sents = get_sents(opt.data_dir, opt.db_info)
    pre_glove = glove2dict(opt.glove_path)
    tokens, embeddings = get_embeddings(sents, pre_glove, opt.n, opt.max_iter)
    st_tokens = set(tokens)

    with open(opt.tuned_glove_path, 'w') as glove_file:
        for token, vec in pre_glove.items():
            if token not in st_tokens:
                line = token + ' ' + np.array2string(vec)[1:-1]
                line = re.sub('\n', ' ', line)
                line = re.sub(' +', ' ', line)
                line = line.strip() + '\n'
                glove_file.write(line)

        for token, vec in zip(tokens, embeddings):
            line = token + ' ' + np.array2string(vec)[1:-1]
            line = re.sub('\n', ' ', line)
            line = re.sub(' +', ' ', line)
            line = line.strip() + '\n'
            glove_file.write(line)
