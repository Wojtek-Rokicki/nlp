"""File with function that runs lstm model
"""
import logging
import os
import pathlib

import torch

from src.data_loading import load_data
import src.model.bert as bert

def single_run_bert(params, bert_version):
    """Runing bert model

        Parameters
       ----------
       params : dictionary
            parameters for running model: 
                    epochs, learning rate, hidden layers dimension, 
                    sequence length,  dataset
        bert version : string
                version of bert model: 'BERT' or 'DistilBERT'
    
       Returns
       -------
       List[dictionary]
            run results
    """
    data_root = os.path.join(pathlib.Path(__file__).parent.parent, 'data')
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")
    logging.info(
        f"Using parameters: sequence_length={params['sequence_length']} "
        f"epochs={params['epochs']} learning_rate={params['learning_rate']}")
    X, y = load_data(data_root, params['datasets'])
    train_dataloader, val_dataloader, test_dataloader = bert.get_preprocessed_dataloaders(X, y, params, bert_version)
    unique_outputs = len(set(y))
    if bert_version == "BERT":
        model = bert.BertClassifier(unique_outputs)
    elif bert_version == "DistilBERT":
        model = bert.DistilBertClassifier(unique_outputs)
    bert.train(model, train_dataloader, val_dataloader, params["learning_rate"], params["epochs"])
    test_results = bert.evaluate(model, test_dataloader)
    test_results = bert.add_parameters_to_test_results(
            test_results, bert_version, params['sequence_length'],
            params['learning_rate'], params['datasets']
        )
    return test_results
