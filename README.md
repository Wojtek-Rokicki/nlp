# Natural Language Processing Project
Comparing BERT with modified LSTM

# How to run?

- Make sure you have downloaded glove pre-trained word vectors from [project website](https://nlp.stanford.edu/data/glove.6B.zip) and put it into data/glove folder.

## TODO:
- [x] Get familiar with the code!
- [x] Test and run LSTM
- [ ] In [data_loading.py](src/data_loading.py) add loaders for datasets
- [ ] Run LSTM on new dataset
- [x] Implement BERT
- [x] Run BERT on new dataset
- [ ] Modify BERT
- [ ] Refactor code
- [ ] Make argparse for LSTM, Roberta, BERT
- [ ] ...

# Notes:
## LSTM types and number of parameters
- LSTM_Layer: 250 114
- LSTM_Single_Cell: 118 018 
- LSTM_POS_Penn: 4 239 618
- LSTM_POS_Universal: 1 531 138