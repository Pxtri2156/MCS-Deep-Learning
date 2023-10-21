import requests
import json
import torch
import os
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

class NaturalQA_Dataset(torch.utils.data.Dataset):
  def __init__(self, encodings):
    self.encodings = encodings
  def __getitem__(self, idx):
    return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
  def __len__(self):
    return len(self.encodings.input_ids)

def read_data(path):  
  # load the json file
  with open(path, 'rb') as f:
    squad = json.load(f)

  contexts = []
  questions = []
  short_answers = []

  for group in squad['data']:
    for passage in group['paragraphs']:
      context = passage['context']
      for qa in passage['qas']:
        question = qa['question']
        for short_answer in qa['short_answers']:
          contexts.append(context)
          questions.append(question)
          short_answers.append(short_answer)

  return contexts, questions, short_answers

def add_end_idx(answers, contexts):
  for answer, context in zip(answers, contexts):
    gold_text = answer['text']
    start_idx = answer['answer_start']
    end_idx = start_idx + len(gold_text)

    # sometimes squad answers are off by a character or two so we fix this
    if context[start_idx:end_idx] == gold_text:
      answer['answer_end'] = end_idx
    elif context[start_idx-1:end_idx-1] == gold_text:
      answer['answer_start'] = start_idx - 1
      answer['answer_end'] = end_idx - 1     # When the gold label is off by one character
    elif context[start_idx-2:end_idx-2] == gold_text:
      answer['answer_start'] = start_idx - 2
      answer['answer_end'] = end_idx - 2     # When the gold label is off by two characters

def add_token_positions(tokenizer, encodings, answers):
  start_positions = []
  end_positions = []
  for i in range(len(answers)):
    start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
    end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))

    # if start position is None, the answer passage has been truncated
    if start_positions[-1] is None:
      start_positions[-1] = tokenizer.model_max_length
    if end_positions[-1] is None:
      end_positions[-1] = tokenizer.model_max_length

  encodings.update({'start_positions': start_positions, 'end_positions': end_positions})


def eval(model, valid_loader, device):
    model.eval()
    acc = []
    for batch in tqdm(valid_loader):
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_true = batch['start_positions'].to(device)
            end_true = batch['end_positions'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)

            start_pred = torch.argmax(outputs['start_logits'], dim=1)
            end_pred = torch.argmax(outputs['end_logits'], dim=1)

            acc.append(((start_pred == start_true).sum()/len(start_pred)).item())
            acc.append(((end_pred == end_true).sum()/len(end_pred)).item())

    acc = sum(acc)/len(acc)
    return acc


def train(model, tokenizer, train_loader, valid_loader, device, args):
    
    optim = AdamW(model.parameters(), lr=args.lr)
    model.to(device)
    model.train()
    for epoch in range(args.epochs):
        loop = tqdm(train_loader, leave=True)
        for batch in loop:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, 
                            start_positions=start_positions, end_positions=end_positions)
            loss = outputs[0]
            loss.backward()
            optim.step()

            loop.set_description(f'Epoch {epoch+1}')
            loop.set_postfix(loss=loss.item())
            
        acc = eval(model, valid_loader, device)
        print("acc: ", acc)
        
    save_path = args.save_path
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Saved at {save_path}")

def arg_parser():
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument("--train_path", 
                        default="/workspace/tripx/MCS/deep_learning/final_qa/data/nq-sub-train-v1.0.1.json", 
                        type=str)
    parser.add_argument("--dev_path", 
                        default="/workspace/tripx/MCS/deep_learning/final_qa/data/nq-sub-val-v1.0.1.json", 
                        type=str)
    parser.add_argument("--test_path", 
                        default="/workspace/tripx/MCS/deep_learning/final_qa/data/nq-sub-val-v1.0.1.json", 
                        type=str)
    parser.add_argument("--save_path", 
                        default="/workspace/tripx/MCS/deep_learning/final_qa/model/v1", 
                        type=str)
    
    # Training
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--epochs", default=2, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)

    # Model 
    parser.add_argument("--model", default="bert-base-uncased", type=str)
    parser.add_argument("--tokenizer", default="bert-base-uncased", type=str)
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--stride", default=256, type=int)
    parser.add_argument("--truncation", default=True, type=bool)
    parser.add_argument("--padding", default=True, type=bool)


    
    return parser.parse_args()

def main(args):
    # Read data 
    print(f"{'='*40}Read data{'='*40}")
    train_contexts, train_questions, train_answers = read_data(args.train_path)
    valid_contexts, valid_questions, valid_answers = read_data(args.dev_path)
    ## add end index of answers
    add_end_idx(train_answers, train_contexts)
    add_end_idx(valid_answers, valid_contexts)
    
    # Convert to tokenizations
    print(f"{'='*40}Convert tokenizations{'='*40}")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train_encodings = tokenizer(train_contexts, train_questions,  
                                max_length=args.max_length, stride=args.stride, 
                                truncation=args.truncation, padding=args.padding)
    valid_encodings = tokenizer(valid_contexts, valid_questions, 
                                max_length=args.max_length, stride=args.stride, 
                                truncation=args.truncation, padding=args.padding)
    ## Convert answer indices to answer tokens
    add_token_positions(tokenizer, train_encodings, train_answers)
    add_token_positions(tokenizer, valid_encodings, valid_answers)
    
    # Load dataset
    print(f"{'='*40}Load data{'='*40}")
    train_dataset = NaturalQA_Dataset(train_encodings)
    valid_dataset = NaturalQA_Dataset(valid_encodings)
    ## Define the dataloaders
    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, 
                              batch_size=args.batch_size)
    
    # Define model
    print(f"{'='*40}Training{'='*40}")
    model = AutoModelForQuestionAnswering.from_pretrained(args.model)
    ## Check on the available device - use GPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Working on {device}')
    
    # Training
    train(model, tokenizer, train_loader, valid_loader, device, args)
    
if __name__ == '__main__':
    args = arg_parser()
    main(args)


