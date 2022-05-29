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

def run_lstm(all_possible_params_lstm):

    results = []
    #LSTM
    embeddings = get_embeddings() # Requires pretrained GloVe embeddings
    for idx, single_params in enumerate(product(*all_possible_params_lstm.values())):
        logging.info(f"Progress for lstm = {idx}/{len(list(product(*all_possible_params_lstm.values())))}")
        single_params_dict = dict(zip(all_possible_params_lstm, single_params))
        run_results = single_run_lstm(single_params_dict, embeddings) # runs for all possible models {LSTM_Layer, LSTM_Single_Cell, LSTM_POS_Penn, LSTM_POS_Universal}
        results.extend(run_results)
        # to do - single params to dict?

    # save_results_to_csv(results, 'lstm')

def run_bert(all_possible_params_bert):
    # BERT
    for idx, single_params in enumerate(product(*all_possible_params_bert.values())):
        logging.info(f"Progress for BERT = {idx}/{len(list(product(*all_possible_params_bert.values())))}")
        single_params_dict = dict(zip(all_possible_params_bert, single_params))
        print(f"all_possible_params_bert: {single_params_dict}")
        run_results = []
        test_results = single_run_bert(single_params_dict, "BERT")
        run_results.append(test_results)
        results = []
        results.extend(run_results)
        save_results_to_csv(results, 'bert')
    
    
def run_distil_bert(all_possible_params_bert):
    # DistilBERT
    for idx, single_params in enumerate(product(*all_possible_params_bert.values())):
        logging.info(f"Progress for DistilBERT = {idx}/{len(list(product(*all_possible_params_bert.values())))}")
        single_params_dict = dict(zip(all_possible_params_bert, single_params))
        run_results = []
        test_results = single_run_bert(single_params_dict, "DistilBERT")
        run_results.append(test_results)
        results=[]
        results.extend(run_results)
        save_results_to_csv(results, 'distilbert')

if __name__ == "__main__":
    start = time.perf_counter()
    download_necessary_nltk_packages()

    results = []
    with open('configs/lstm_config.json', 'r') as lstm_fp, open('configs/bert_config.json', 'r') as bert_fp: #, open('configs/roberta_config.json', 'r') as roberta_fp:
        all_possible_params_lstm = json.load(lstm_fp)
        all_possible_params_bert = json.load(bert_fp)
    
    run_bert(all_possible_params_bert)
    run_distil_bert(all_possible_params_bert)
    run_lstm(all_possible_params_lstm)

    end = time.perf_counter()
    print(f"Exec time: {end - start}")
