# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 12:25:15 2022

@author: Rohit Gandikota
"""

import pandas as pd
import numpy as np
import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
import os

# Downloading tokenizer file for model using python as file is huge
import requests
import json
name = 'tokenizer.json'
url = f'https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/raw/main/{name}'
resp = requests.get(url, verify=False)
dict_ = resp.json()
json_object = json.dumps(dict_, indent = 4)
with open(f'D:\\Projects\\bhoonidhi\\bert-large-uncased-whole-word-masking-finetuned-squad\\{name}', "w") as outfile:
    outfile.write(json_object)
    
url = 'https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/raw/main/vocab.txt'
resp = requests.get(url,verify=False)
dict_ = resp.text
with open(f'D:\\Projects\\bhoonidhi\\bert-large-uncased-whole-word-masking-finetuned-squad\\vocab.txt', "w", encoding="utf-8") as outfile:
    outfile.write(dict_)
    
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')


question = 'When was kompsat-3 launched?'
text = '''KOMPSAT-3 is a high performance remote sensing satellite, which provides 0.7 m GSD
panchromatic image and 2.8 m GSD multi-spectral image data for various applications.
KOMPSAT-3 was launched into a sun synchronous low Earth orbit on the 18th of May, 2012
and the life time of more than 7 years is expected.'''
input_ids = tokenizer.encode(question, text)
print("The input has a total of {} tokens.".format(len(input_ids)))
tokens = tokenizer.convert_ids_to_tokens(input_ids)
for token, id in zip(tokens, input_ids):
    print('{:8}{:8,}'.format(token,id))
    
#first occurence of [SEP] token
sep_idx = input_ids.index(tokenizer.sep_token_id)
print("SEP token index: ", sep_idx)#number of tokens in segment A (question) - this will be one more than the sep_idx as the index in Python starts from 0
num_seg_a = sep_idx+1
print("Number of tokens in segment A: ", num_seg_a)#number of tokens in segment B (text)
num_seg_b = len(input_ids) - num_seg_a
print("Number of tokens in segment B: ", num_seg_b)#creating the segment ids
segment_ids = [0]*num_seg_a + [1]*num_seg_b#making sure that every input token has a segment id
assert len(segment_ids) == len(input_ids)


#token input_ids to represent the input and token segment_ids to differentiate our segments - question and text
output = model(torch.tensor([input_ids]),  token_type_ids=torch.tensor([segment_ids]))

#tokens with highest start and end scores
answer_start = torch.argmax(output.start_logits)
answer_end = torch.argmax(output.end_logits)
if answer_end >= answer_start:
    answer = " ".join(tokens[answer_start:answer_end+1])
else:
    print("I am unable to find the answer to this question. Can you please ask another question?")
    
print("\nQuestion:\n{}".format(question.capitalize()))
print("\nAnswer:\n{}.".format(answer.capitalize()))
