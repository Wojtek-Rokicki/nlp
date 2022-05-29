
import torch
import torch.nn as nn
from torch.optim import Adam

from transformers import BertModel, DistilBertModel
from transformers import BertTokenizer, DistilBertTokenizer

import numpy as np
import pandas as pd

from tqdm import tqdm
import logging

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch.nn.functional as F

class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 5)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer

class DistilBertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(DistilBertClassifier, self).__init__()

        self.bert = DistilBertModel.from_pretrained('distilbert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 6)
        self.relu = nn.ReLU()


    def forward(self, input_id, mask):

        outputs = self.bert(input_ids= input_id, attention_mask=mask,return_dict=True)
        last_hidden_state = outputs[0]
        pooled_output = last_hidden_state[:,0,:]
        
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        result = F.log_softmax(final_layer, dim = 1)

        return result

class Dataset(torch.utils.data.Dataset):

    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y


def train(model, train_dataloader, val_dataloader, learning_rate, epochs):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()    
    optimizer = Adam(model.parameters(), lr= learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):

        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):

            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)
            
            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()
            
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
        
        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

            for val_input, val_label in val_dataloader:

                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()
                
                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc
        
        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_dataloader.dataset): .3f} \
            | Train Accuracy: {total_acc_train / len(train_dataloader.dataset): .3f} \
            | Val Loss: {total_loss_val / len(val_dataloader.dataset): .3f} \
            | Val Accuracy: {total_acc_val / len(val_dataloader.dataset): .3f}')

def evaluate(model, test_dataloader):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if use_cuda:

        model = model.cuda()

    test_lab_vec, test_pred_vec = [], []
    with torch.no_grad():
        for test_input, test_label in test_dataloader:

            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            test_pred_vec.extend(list(output.cpu().argmax(dim=1)))
            test_lab_vec.extend(list(test_label.cpu()))

    accuracy = accuracy_score(y_true=test_lab_vec, y_pred=test_pred_vec)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true=test_lab_vec, y_pred=test_pred_vec, beta=1,
                                                        average='weighted')
    logging.info(
        f'\nAccuracy: {accuracy:.4f}'
        f'\nAverage precision score: {prec:.4f}'
        f'\nAverage recall score: {rec:0.4f}'
        f'\nAverage f1-recall score: {f1:0.4f}'
    )

    return {
        "accuracy": accuracy,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }

def add_parameters_to_test_results(test_results, model_name, sequence_length,
                                     learning_rate, dataset):

    test_results["model"] = model_name
    test_results["sequence_length"] = sequence_length
    # test_results["embedding_size"] = embedding_size
    # test_results["epochs"] = epochs
    test_results["learning_rate"] = learning_rate
    test_results["dataset"] = dataset

    return test_results


def preprocess_text(X, y, params, bert_version):
    """ Tokenizes texts and encodes labels.

    Returns:
        texts:
            List of sentences embeddings.
        labels: 
            List of numerically encoded labels of sentences.
    """

    if bert_version == "BERT":
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    elif bert_version == "DistilBERT":
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')

    label_encoder = preprocessing.LabelEncoder()
    labels = label_encoder.fit_transform(y)
    texts = [tokenizer(text, padding='max_length', max_length = params["sequence_length"], truncation=True, return_tensors="pt") for text in X]
    return texts, labels

def split_into_datasets(X, y, params):
    """ Splits inputs and labels into train, validation and test subsets.

        Splits dataset with ratios specified in params.

        Returns:
            List of three Dataset objects of train, validation and test subsets.
    """

    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=params['train_ratio'], random_state=7)
    split_idx = int(params['val_ratio'] / (params['val_ratio'] + params['test_ratio']) * len(x_test))
    x_val, x_test, y_val, y_test = x_test[:split_idx], x_test[split_idx:], y_test[:split_idx], y_test[split_idx:]

    return Dataset(x_train, y_train), Dataset(x_val, y_val), Dataset(x_test, y_test)

def get_preprocessed_dataloaders(X, y, params, bert_version):
    """ Gets dataloaders of train, validation and test subsets.

        Returns:
            List of three DataLoaders objects of train, validation and test subsets.
    """

    texts, labels = preprocess_text(X, y, params, bert_version)

    train, val, test = split_into_datasets(texts, labels, params)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=params["batch_size"], shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=params["batch_size"])
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=params["batch_size"])

    return train_dataloader, val_dataloader, test_dataloader

    