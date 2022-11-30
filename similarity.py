import csv

import nlpaug.augmenter.word as naw
import nltk
import torch
import numpy as np

import torch.nn as nn

nltk.download('punkt')
from nltk.tokenize import sent_tokenize

import random
import re
import time

import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--output_file', type=str, required=True)
# parser.add_argument('--input_file', type=str, required=True)
# parser.add_argument('--model_path', type=str, required=True)
#
# args = parser.parse_args()

input_file = 'dataset/dataset.csv'
model_path = 'bert-base-uncased'

list2 = [1, 2]

pattern = r'\[.*?\]'
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
print(device)

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('bert-base-uncased', device=device)
model.to(device)
cos = nn.CosineSimilarity(dim=1, eps=1e-6)
cos.to(device)

# for some reason, loading the embedding first time taking a lot of time. So to avoid that I am just computing a random embedding at the beginning
sentences = ["a is the first letter", "a cat is bigger than a bird"]
sentence_embeddings = model.encode(sentences)
sim = cos(torch.Tensor(sentence_embeddings[0].reshape(-1,1)), torch.Tensor(sentence_embeddings[1].reshape(-1,1)))


with open(input_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    next(csv_reader)
    flag_c = False
    start = time.time()
    for row in csv_reader:
        sentences = [row[1], row[2]]
        sentence_embeddings = model.encode(sentences)
        sim = cos(torch.Tensor(sentence_embeddings[0].reshape(-1,1)), torch.Tensor(sentence_embeddings[1].reshape(-1,1)))
        # sim = cosine_similarity([sentence_embeddings[0]], sentence_embeddings[1:])
        end = time.time()
        print(end - start)
csv_file.close()
