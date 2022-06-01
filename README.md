# Natural Language Processing Project
Comparing BERTs with modified LSTMs

# Remarks:
## LSTMs (& RoBERTa)
Initial implementation was done by:
- Adam Kapica
- Piotr Kramek

Pretrained [glove word embeddings](https://nlp.stanford.edu/data/glove.6B.zip)

## BERT & DistilBERT
Resources for implementation:
- [Text Classification with BERT by Ruben Winastwan](https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f)

# TODO:
- [x] Get familiar with the code!
- [x] Test and run LSTM
- [x] Add loaders for all datasets
- [x] Implement BERT
- [x] Test and run BERT
- [x] Code evaluation for BERT
- [x] Add DistilBERT
- [x] Compare models
- [x] Refactor code
- [x] Analyze and document results
- [x] Write report

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
