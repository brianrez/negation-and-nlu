import torch
with open('val_preds.p', 'rb') as f:
    # predictions = pickle.load(f)
    preds = torch.load('val_preds.p')

print(preds)
print(len(preds['wsc']['preds']))

import json
from sklearn.metrics import classification_report, accuracy_score
val = []
with open('../../content/exp/tasks/data/wsc/val.jsonl') as f:
    for line in f:
        data = json.loads(line)
        val.append(data)

labels = [1 if instance['label'] is True else 0 for instance in val]

print(accuracy_score(labels, preds['wsc']['preds']))

print(classification_report(labels, preds['wsc']['preds']))

import csv

data = []
# Replace 'file_path.tsv' with your TSV file's path
with open('../../annotations/important-unimportant/wsc.tsv', 'r', newline='') as file:
    reader = csv.reader(file, delimiter='\t')
    for row in reader:
        data.append(row)

data = data[1:]

indices = [row[1] for row in data]

new_labels = []
new_preds = []
for index in indices:
    new_labels.append(labels[int(index)])
    new_preds.append(preds['wsc']['preds'][int(index)])
print("W/ negation")
print(accuracy_score(new_labels, new_preds))
print(classification_report(new_labels, new_preds))

new_labels = []
new_preds = []
indexs = [i for i in range(len(labels)) if i not in indices]
for index in indexs:
    new_labels.append(labels[int(index)])
    new_preds.append(preds['wsc']['preds'][int(index)])
print("W/O negation")
print(accuracy_score(new_labels, new_preds))
print(classification_report(new_labels, new_preds))

