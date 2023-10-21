import requests
import json
import torch
import os
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

from transformers import BertForQuestionAnswering, BertTokenizerFast

model_path = '/workspace/tripx/MCS/deep_learning/final_qa/model/bert'
distil_bert = '/workspace/tripx/MCS/deep_learning/final_qa/model/distibert'
pre_train_model =  'bert-base-uncased'

model = BertForQuestionAnswering.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Working on {device}')

model = model.to(device)

def get_prediction(model, context, question):
  inputs = tokenizer.encode_plus(question, context, return_tensors='pt').to(device)
  outputs = model(**inputs)
  answer_start = torch.argmax(outputs[0])  
  answer_end = torch.argmax(outputs[1]) + 1 
  answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
  
  return answer


import json
path='/workspace/tripx/MCS/deep_learning/final_qa/data/nq-sub-val-v1.0.1.json'

f = open(path)
data_test = json.load(f)
print(type(data_test))
f.close()

# context_qa = data_test['data'][2]['paragraphs'][0]['context']
# question = data_test['data'][2]['paragraphs'][0]['qas'][0]['question']
# true_answer = data_test['data'][2]['paragraphs'][0]['qas'][0]['short_answers']

# context_qa = 'The Imperial Palace is the main residence of the Emperor of Japan. It is a large park-like area located in the Chiyoda district of the Chiyoda ward of Tokyo and contains several buildings including the Fukiage Palace (吹上御所, Fukiage gosho) where the Emperor has his living quarters, the main palace (宮殿, Kyūden) where various ceremonies and receptions take place, some residences of the Imperial Family, an archive, museums and administrative offices.'
# question = "Who lives in the imperial palace in tokyo?"
# true_answer = "The Imperial Family"

question="how i.met your mother who is the mother"
true_answer = 'McConnell'
context_qa='Human oddler </Li> <Li> Early child tract over'

answer = get_prediction(model, context_qa, question)

print("Predict Answer: ", answer)
print("True Answer: ", true_answer)