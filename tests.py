import json
import os
import pathlib

from src.data_loading import load_data
import src.model.bert as bert
from tqdm import tqdm

def params_single_list_uzip(params):
    for x in params:
        if len(params[x]) == 1:
            params[x] = params[x][0]
    return params

def bert_text_preprocessing_test(params, data_root):
    params = params_single_list_uzip(params)
    for dataset in tqdm(params["datasets"]):
        X, y = load_data(data_root, dataset)
        train_dataloader, val_dataloader, test_dataloader = bert.get_preprocessed_dataloaders(X, y, params)
    return True

def bert_evaluation_test(params, data_root):
    model = bert.BertClassifier()
    params = params_single_list_uzip(params)
    X, y = load_data(data_root, params["datasets"][0])
    train_dataloader, val_dataloader, test_dataloader = bert.get_preprocessed_dataloaders(X, y, params)
    bert.evaluate(model, test_dataloader)
    return True

if __name__ == "__main__":
    data_root = os.path.join(pathlib.Path(__file__).parent.parent, 'nlp/data')
    f = open('configs/bert_config.json', 'r')
    params = json.load(f)

    # if bert_text_preprocessing_test(params, data_root) == True:
    #     print("BERT text preprocessing test passed sucessfully!")

    if bert_evaluation_test(params, data_root) == True:
        print("BERT text preprocessing test passed sucessfully!")

    f.close()
