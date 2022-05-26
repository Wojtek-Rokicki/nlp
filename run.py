import json
import logging
import time
from itertools import product

from src.data_loading import download_necessary_nltk_packages
from src.single_run_lstm import single_run_lstm
from src.single_run_bert import single_run_bert
from src.single_run_roberta import single_run_roberta
from src.utils import save_results_to_csv, get_embeddings

logging.basicConfig(format='%(levelname)s - %(asctime)s - %(message)s', level=logging.INFO)

if __name__ == "__main__":
    start = time.perf_counter()
    download_necessary_nltk_packages()

    results = []
    with open('configs/lstm_config.json', 'r') as lstm_fp, open('configs/bert_config.json', 'r') as bert_fp: #, open('configs/roberta_config.json', 'r') as roberta_fp:
        all_possible_params_lstm = json.load(lstm_fp)
        all_possible_params_bert = json.load(bert_fp)
        #all_possible_params_roberta = json.load(roberta_fp)

    #LSTM
    embeddings = get_embeddings() # Requires pretrained GloVe embeddings
    for idx, single_params in enumerate(product(*all_possible_params_lstm.values())):
        logging.info(f"Progress for lstm = {idx}/{len(list(product(*all_possible_params_lstm.values())))}")
        single_params_dict = dict(zip(all_possible_params_lstm, single_params))
        run_results = single_run_lstm(single_params_dict, embeddings) # runs for all possible models {LSTM_Layer, LSTM_Single_Cell, LSTM_POS_Penn, LSTM_POS_Universal}
        results.extend(run_results)

    save_results_to_csv(results, 'lstm')

    # BERT
    results = []
    for idx, single_params in enumerate(product(*all_possible_params_bert.values())):
        logging.info(f"Progress for BERT = {idx}/{len(list(product(*all_possible_params_bert.values())))}")
        single_params_dict = dict(zip(all_possible_params_bert, single_params))
        run_results = single_run_bert(single_params_dict)
        results.extend(run_results)


    save_results_to_csv(results, 'bert')

    # # RoBERTa
    # results = []
    # for idx, single_params in enumerate(product(*all_possible_params_roberta.values())):
    #     logging.info(f"Progress for roberta {idx}/{len(list(product(*all_possible_params_roberta.values())))}")
    #     single_params_dict = dict(zip(all_possible_params_roberta, single_params))
    #     run_results = single_run_roberta(single_params_dict)
    #     results.extend(run_results)

    # save_results_to_csv(results, "roberta")

    end = time.perf_counter()
    print(f"Exec time: {end - start}")
