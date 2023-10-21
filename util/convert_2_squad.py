import requests
import json
import torch
import os
from tqdm import tqdm

# train_path = '/workspace/tripx/MCS/deep_learning/squad_data_split/train-v2.0.json'
# dev_path = '/workspace/tripx/MCS/deep_learning/squad_data_split/dev-v2.0.json'


# def read_data(path):  
#   # load the json file
#   with open(path, 'rb') as f:
#     squad = json.load(f)

#   contexts = []
#   questions = []
#   answers = []

#   for group in squad['data']:
#     for passage in group['paragraphs']:
#       context = passage['context']
#       for qa in passage['qas']:
#         question = qa['question']
#         for answer in qa['answers']:
#           contexts.append(context)
#           questions.append(question)
#           answers.append(answer)

#   return contexts[:50], questions[:50], answers[:50]


# train_contexts, train_questions, train_answers = read_data(train_path)
# valid_contexts, valid_questions, valid_answers = read_data(dev_path)


# def add_end_idx(answers, contexts):
#   for answer, context in zip(answers, contexts):
#     gold_text = answer['text']
#     start_idx = answer['answer_start']
#     end_idx = start_idx + len(gold_text)

#     # sometimes squad answers are off by a character or two so we fix this
#     if context[start_idx:end_idx] == gold_text:
#       answer['answer_end'] = end_idx
#     elif context[start_idx-1:end_idx-1] == gold_text:
#       answer['answer_start'] = start_idx - 1
#       answer['answer_end'] = end_idx - 1     # When the gold label is off by one character
#     elif context[start_idx-2:end_idx-2] == gold_text:
#       answer['answer_start'] = start_idx - 2
#       answer['answer_end'] = end_idx - 2     # When the gold label is off by two characters

# add_end_idx(train_answers, train_contexts)
# add_end_idx(valid_answers, valid_contexts)

# print(train_contexts[0])
# print(train_questions[0])
# print(train_answers[0])

import json
import pandas as pd

train_path = '/workspace/tripx/MCS/deep_learning/natural-question-answering/simplified-nq-train.jsonl'
test_path = '/workspace/tripx/MCS/deep_learning/natural-question-answering/simplified-nq-test.jsonl'

train_data = pd.read_json(path_or_buf=train_path, lines=True)