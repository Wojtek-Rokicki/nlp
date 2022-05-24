import csv
import os

import nltk
import numpy as np
import tqdm


def load_data(data_root, dataset):
    if dataset == 'sms_spam':
        return load_sms_spam_ds(data_root)
    elif dataset == 'disaster_tweets':
        return load_tweets_ds(data_root)
    elif dataset == 'emotions':
        return load_emotions_ds(data_root)
    elif dataset == 'imdb':
        return load_imdb_ds(data_root)
    elif dataset == 'bbc':
        return load_bbc_ds(data_root)


def load_sms_spam_ds(data_root):
    texts, labels = [], []
    with open(os.path.join(data_root, 'datasets/sms_spam', 'sms_spam_dataset')) as f:
        for line in f:
            labels.append(line.split()[0].strip()), texts.append(' '.join(line.split()[1:]).strip())
    return texts, labels

def load_tweets_ds(data_root):
    texts, labels = [], []
    with open(os.path.join(data_root, 'datasets/disaster_tweets', 'train.csv')) as csv_file:
        for row in csv.DictReader(csv_file, delimiter=','):
            texts.append(row['text']), labels.append(int(row['target']))
    with open(os.path.join(data_root, 'datasets/disaster_tweets', 'test.csv')) as csv_file:
        test_data = {row['id']: row['text'] for row in csv.DictReader(csv_file, delimiter=',')}
    with open(os.path.join(data_root, 'datasets/disaster_tweets', 'sample_submission.csv')) as csv_file:
        test_labels = {row['id']: row['target'] for row in csv.DictReader(csv_file, delimiter=',')}
    for id in test_data.keys():
        texts.append(test_data[id]), labels.append(int(test_labels[id]))
    return texts, labels

def load_emotions_ds(data_root):
    texts, labels = [], []
    with open(os.path.join(data_root, 'datasets/emotions', 'train.txt')) as f:
        for line in f:
            texts.append(line.split(';')[0].strip()), labels.append(line.split(';')[1].strip())
    with open(os.path.join(data_root, 'datasets/emotions', 'val.txt')) as f:
        for line in f:
            texts.append(line.split(';')[0].strip()), labels.append(line.split(';')[1].strip())
    with open(os.path.join(data_root, 'datasets/emotions', 'test.txt')) as f:
        for line in f:
            texts.append(line.split(';')[0].strip()), labels.append(line.split(';')[1].strip())
    return texts, labels

def load_imdb_ds(data_root):
    texts, labels = [], []
    with open(os.path.join(data_root, 'datasets/imdb', 'imdb_dataset.csv')) as csv_file:
        for row in csv.DictReader(csv_file, delimiter=','):
            texts.append(row['review']), labels.append((row['sentiment']))
    return texts, labels

def load_bbc_ds(data_root):
    texts, labels = [], []
    with open(os.path.join(data_root, 'datasets/bbc', 'bbc_articles.csv')) as csv_file:
        for row in csv.DictReader(csv_file, delimiter=','):
            texts.append(row['text']), labels.append(row['category'])
    return texts, labels

# https://nlp.stanford.edu/projects/glove/
def load_glove_embedding_vec(data_root, dim):
    embedding_index = dict()
    with open(os.path.join(data_root, f"glove/glove.6B.{dim}d.txt")) as f:
        for line in tqdm.tqdm(f, "Reading GloVe"):
            values = line.split()
            word = values[0]
            embedding_index[word] = np.asarray(values[1:], dtype='float32')
    return embedding_index


def download_necessary_nltk_packages():
    nltk.download(['averaged_perceptron_tagger', 'universal_tagset', 'tagsets'])
