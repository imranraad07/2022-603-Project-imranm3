import csv

import nlpaug.augmenter.word as naw
import nltk
import torch

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
aug_insert = naw.ContextualWordEmbsAug(model_path=model_path, action="insert", aug_max=1, device=device)
aug_substitute = naw.ContextualWordEmbsAug(model_path=model_path, action="substitute", aug_max=1, device=device)

with open(input_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    next(csv_reader)
    flag_c = False
    start = time.time()
    for row in csv_reader:
        sentence = row[1]
        max_try = 0
        while max_try < 10:
            try:
                max_try = max_try + 1
                augmented_text = sentence
                for operations in range(5):
                    choice_1 = random.choice(list2)
                    if choice_1 == 1:
                        augmented_text = aug_insert.augment(augmented_text)
                    elif choice_1 == 2:
                        augmented_text = aug_substitute.augment(augmented_text)
            except Exception as e:
                print("Exception occurred", e)
        end = time.time()
        print(end - start)
csv_file.close()
