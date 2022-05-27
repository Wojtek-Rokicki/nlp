import logging
import os
import pathlib

import torch
import torch.nn as nn

from src.data_loading import load_data
from src.test import test
from src.train import train
from src.utils import Selector, get_model, get_embedding_vectors, prepare_data_loaders_and_tokenizer, \
    add_parameters_to_test_results, count_parameters
from src.utils import save_results_to_csv



def single_run_lstm(params, embeddings, all_models=True, model_idx=0):
    data_root = os.path.join(pathlib.Path(__file__).parent.parent, 'data')
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")
    logging.info(
        f"Using parameters: sequence_length={params['sequence_length']} embedding_size={params['embedding_size']} "
        f"epochs={params['epochs']} learning_rate={params['learning_rate']} padding={params['padding']}")
    X, y = load_data(data_root, params['datasets'])
    train_loader, val_loader, test_loader, tokenizer, output_size = prepare_data_loaders_and_tokenizer(X, y, params)
    embedding_matrix = get_embedding_vectors(tokenizer, params['embedding_size'], embeddings)
    vocab_size = len(tokenizer.word_index) + 1

    run_results = []
    if all_models == True:
        for model_idx in range(len(Selector)):
            test_results = single_model_lstm_run(model_idx, data_root, device, train_loader, val_loader, test_loader, tokenizer,output_size, embedding_matrix, vocab_size, params )

            run_results.append(test_results)
            results = []
            results.extend(run_results)
            save_results_to_csv(results, 'lstm')
    else:
        test_results = single_model_lstm_run(moddel_idx, data_root, device, train_loader, val_loader, test_loader, tokenizer,output_size, embedding_matrix, vocab_size, params )

        run_results.append(test_results)
        results = []
        results.extend(run_results)
        save_results_to_csv(results, 'lstm')

    return run_results

def single_model_lstm_run(model_idx, data_root, device, train_loader, val_loader, test_loader, tokenizer, output_size, embedding_matrix, vocab_size, params):
    model_name = Selector(model_idx).name
    logging.info(f"name of the model: {model_name}")

    model = get_model(Selector(model_idx), vocab_size, output_size, embedding_matrix, params['embedding_size'],
                        params['hidden_dim'], device, params['drop_prob'], tokenizer, params['sequence_length'])
    no_params = count_parameters(model)
    logging.info(f"model has {no_params} trainable parameters")
    criterion, optimizer = nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

    train_stats = \
        train(model, params['epochs'], train_loader, val_loader, device, optimizer, criterion)
    test_results = test(model, test_loader, device, criterion)

    test_results = add_parameters_to_test_results(
        test_results, model_name, params['sequence_length'], params['embedding_size'],
        train_stats['epoch_min_loss'], params['learning_rate'], params['padding'], params['datasets']
    )

    return test_results

