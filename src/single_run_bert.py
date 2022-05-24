import logging
import os
import pathlib

import torch
import src.model.bert as bert


def single_run_bert(params):
    data_root = os.path.join(pathlib.Path(__file__).parent.parent, 'data')
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")
    logging.info(
        f"Using parameters: sequence_length={params['sequence_length']} "
        f"epochs={params['epochs']} learning_rate={params['learning_rate']}")
    train_dataloader, val_dataloader, test_dataloader = bert.get_preprocessed_dataloaders(params['dataset'])
    model = bert.BertClassifier()
    bert.train(model, train_dataloader, val_dataloader, params["learning_rate"], params["epochs"])
    bert.evaluate(model, test_dataloader)