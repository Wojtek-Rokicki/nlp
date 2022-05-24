# Natural Language Processing Project
Comparing BERT with modified LSTMs

# Remarks:
## LSTMs & RoBERTa
Initial implementation was done by:
- Adam Kapica
- Piotr Kramek

Pretrained [glove word embeddings](https://nlp.stanford.edu/data/glove.6B.zip)

## BERT
Resources for implementation:
- [Text Classification with BERT by Ruben Winastwan](https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f)

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
## Prerequisities:
- Resources to put under data directory (https://drive.google.com/drive/folders/1dWAChGX-tV9eeJ9gldqEXKMXLk3J4gtf?usp=sharing)
## Model types and number of parameters
- LSTM_Layer: ~250k
- LSTM_Single_Cell: ~100k
- LSTM_POS_Penn: ~4M
- LSTM_POS_Universal: ~1.5M
- BERT: ~110M
- DistilBERT: ~65M
## Knowledge resources:
- [Understanding LSTMs by Christopher Olah](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Transformers by Ketan Doshi](https://towardsdatascience.com/transformers-explained-visually-part-1-overview-of-functionality-95a6dd460452)