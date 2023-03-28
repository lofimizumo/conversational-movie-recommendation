from model.recModel import LitRecModel
import pandas as pd
from util.dataset import *
from collections import OrderedDict

dataset = MoviePlotDataset('plot', 'question')
ret=next(iter(dataset))

model = LitRecModel.load_from_checkpoint("lightning_logs/version_0/checkpoints/epoch=1-step=40.ckpt")

model.eval()

movie_dict=read_plots_from_csv('data/movie_1000_clean.csv')
names=movie_dict.keys()
idx=list(range(len(names)))
idx_name_dict=OrderedDict(zip(idx,names))
question = 'can you recommend a movie about a man seeking supernatural power'
ret = model.inference(question)
print(f'Movie:{idx_name_dict[int(ret)]}')

question = 'can you recommend a movie about dinosaur'
ret = model.inference(question)
print(f'Movie:{idx_name_dict[int(ret)]}')

#%%
from transformers import BertModel, BertTokenizer
import torch

model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

input_text = "This is an example input sentence."

input_tokens = tokenizer.encode(input_text, add_special_tokens=True, return_tensors='pt')
input_text2 = "can you recommend a movie about dinosaur"

input_tokens2 = tokenizer.encode(input_text2, add_special_tokens=True, return_tensors='pt')
tokens=[input_tokens,input_tokens2]
tokens=torch.cat(tokens)
with torch.no_grad():
    encoded_input = model(tokens)[0]

# Print the encoded representation of the input text
print(encoded_input)

#%%
questions=['can you recommend a movie about dinosaur',

'can you recommend a movie about alien fighting human',

'can you recommend a movie about love story',

'can you recommend a movie about ghost and spirit',

'can you recommend a movie about animals',

'can you recommend a movie about war between countries',

'can you recommend a movie about kids and families',

'can you recommend a movie about kids tring to save the alien',

'can you recommend a movie about animals fighting each other for power',

'can you recommend a movie about a thrill adventure']
questions