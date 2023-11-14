import torch
with open('./preds/commonsenseqa/ch_large_1e-5.pt', 'rb') as f:
    # predictions = pickle.load(f)
    preds = torch.load(f)

print(preds)
print(len(preds['commonsenseqa']['preds']))

import json
from sklearn.metrics import classification_report, accuracy_score
val = []
with open('./data/commonsenseqa/ch/val.jsonl') as f:
    for line in f:
        data = json.loads(line)
        val.append(data)

label2id = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
labels = [label2id[instance['answerKey']] for instance in val]

print(accuracy_score(labels, preds['commonsenseqa']['preds']))

print(classification_report(labels, preds['commonsenseqa']['preds']))

import csv

data = []
# Replace 'file_path.tsv' with your TSV file's path
with open('./annotations/important-unimportant/commonsenseqa.tsv', 'r', newline='') as file:
    reader = csv.reader(file, delimiter='\t')
    for row in reader:
        data.append(row)

data = data[1:]

indices = []
for i in range(len(val)):
    for j in range(len(data)):
        if val[i]['question'].strip() == data[j][2].strip():
            indices.append(i)
            break

assert len(indices) == len(data), f'{len(indices)} != {len(data)}'


new_labels = []
new_preds = []
for index in indices:
    new_labels.append(labels[int(index)])
    new_preds.append(preds['commonsenseqa']['preds'][int(index)])
print("W/ negation")
print(accuracy_score(new_labels, new_preds))
print(classification_report(new_labels, new_preds))

new_labels = []
new_preds = []
indexs = [i for i in range(len(labels)) if i not in indices]
for index in indexs:
    new_labels.append(labels[int(index)])
    new_preds.append(preds['commonsenseqa']['preds'][int(index)])
print("W/O negation")
print(accuracy_score(new_labels, new_preds))
print(classification_report(new_labels, new_preds))
