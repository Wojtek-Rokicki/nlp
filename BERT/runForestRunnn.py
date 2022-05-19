from transformers import BertTokenizer
import json

tokenizer = BertTokenizer.from_pretrained('distilbert-base-cased') # Maybe array for different BERTs

example_text = 'I will watch Memento tonight'
bert_input = tokenizer(example_text,padding='max_length', max_length = 10, 
                       truncation=True, return_tensors="pt")


print(bert_input['input_ids'])