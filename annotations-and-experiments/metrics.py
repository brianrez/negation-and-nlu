from utils import Jsonl
import os
from sklearn.metrics import accuracy_score, classification_report
import csv
import torch

class metrics:
    def __init__(self, task, model, setting, lr):
        self.task = task
        self.model = model
        self.setting = setting
        self.lr = lr

        self.data = None
        self.preds = None
        self.import_neg_indices = []
        self.not_import_neg_indices = []
        self.neg_indices = []
        self.non_neg_indices = []

        self.results_str = ''

    def write_to_text(self):
        pth = f'./results/{self.task}/'
        if not os.path.exists(pth):
            os.makedirs(pth)

        with open(f'./results/{self.task}/{self.setting}_{self.model}_{self.lr}.txt', 'w') as f:
            f.write(self.results_str)

    def read_pred_data(self):
        self.data = Jsonl().read(f'./data/{self.task}/or/val.jsonl')
        self.preds = torch.load(f'./preds/{self.task}/{self.setting}_{self.model}_{self.lr}.pt')

    def commonsenseqa(self):
        label2id = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
        labels = [label2id[instance['answerKey']] for instance in self.data]
        # Get all 
        self.results_str += f'All data: {accuracy_score(labels, self.preds["commonsenseqa"]["preds"])}\n'
        self.results_str += f'All data: {classification_report(labels, self.preds["commonsenseqa"]["preds"])}\n'
        
        # Get indices of important and not important negations
        import csv
        data = []
        # Replace 'file_path.tsv' with your TSV file's path
        with open('./annotations/important-unimportant/commonsenseqa.tsv', 'r', newline='') as file:
            reader = csv.reader(file, delimiter='\t')
            for row in reader:
                data.append(row)

        data = data[1:]
        
        for i in range(len(self.data)):
            for j in range(len(data)):
                if self.data[i]['question'].strip() == data[j][2].strip():
                    if data[j][6] == "Yes":
                        self.import_neg_indices.append(i)
                    else:
                        self.not_import_neg_indices.append(i)
                    self.neg_indices.append(i)
                    break

        assert len(self.neg_indices) == len(data), f'{len(self.neg_indices)} != {len(data)}'

        def index_to_metrics(indices, setting):
            new_labels = []
            new_preds = []
            for index in indices:
                new_labels.append(labels[int(index)])
                new_preds.append(self.preds['commonsenseqa']['preds'][int(index)])
            self.results_str += f'{setting}: {accuracy_score(new_labels, new_preds)}\n'
            self.results_str += f'{setting}: {classification_report(new_labels, new_preds)}\n'

        # Get indices of non-negations
        self.non_neg_indices = [i for i in range(len(labels)) if i not in self.neg_indices]
        index_to_metrics(self.neg_indices, 'W/ negation')
        index_to_metrics(self.non_neg_indices, 'W/O negation')
        index_to_metrics(self.import_neg_indices, 'W/ negation and important')
        index_to_metrics(self.not_import_neg_indices, 'W/ negation and not important')

    def wsc(self):
        labels = [1 if instance['label'] is True else 0 for instance in self.data]
        # Get all 
        self.results_str += f'All data: {accuracy_score(labels, self.preds["wsc"]["preds"])}\n'
        self.results_str += f'All data: {classification_report(labels, self.preds["wsc"]["preds"])}\n'
        
        # Get indices of important and not important negations
        import csv
        data = []
        # Replace 'file_path.tsv' with your TSV file's path
        with open('./annotations/important-unimportant/wsc.tsv', 'r', newline='') as file:
            reader = csv.reader(file, delimiter='\t')
            for row in reader:
                data.append(row)

        data = data[1:]
        
        for row in data:
            if row[6] == "Yes":
                self.import_neg_indices.append(row[1])
            else:
                self.not_import_neg_indices.append(row[1])
            self.neg_indices.append(row[1])
        self.non_neg_indices = [i for i in range(len(labels)) if i not in self.neg_indices]

        assert len(self.neg_indices) == len(data)


        def index_to_metrics(indices, setting):
            new_labels = []
            new_preds = []
            for index in indices:
                new_labels.append(labels[int(index)])
                new_preds.append(self.preds['wsc']['preds'][int(index)])
            self.results_str += f'{setting}: {accuracy_score(new_labels, new_preds)}\n'
            self.results_str += f'{setting}: {classification_report(new_labels, new_preds)}\n'

        # Get indices of non-negations
        self.non_neg_indices = [i for i in range(len(labels)) if i not in self.neg_indices]
        index_to_metrics(self.neg_indices, 'W/ negation')
        index_to_metrics(self.non_neg_indices, 'W/O negation')
        if len(self.import_neg_indices) > 0:
            index_to_metrics(self.import_neg_indices, 'W/ negation and important')
        else:
            self.results_str += 'W/ negation and important: No data\n'
        index_to_metrics(self.not_import_neg_indices, 'W/ negation and not important')

    def wic(self):
        labels = [1 if instance['label'] is True else 0 for instance in self.data]

        # Get all
        self.results_str += f'All data: {accuracy_score(labels, self.preds["wic"]["preds"])}\n'
        self.results_str += f'All data: {classification_report(labels, self.preds["wic"]["preds"])}\n'

        # Get indices of important and not important negations
        import csv
        data = []
        # Replace 'file_path.tsv' with your TSV file's path
        with open('./annotations/important-unimportant/wic.tsv', 'r', newline='') as file:
            reader = csv.reader(file, delimiter='\t')
            for row in reader:
                data.append(row)
        
        data = data[1:]

        for row in data:
            if row[8] == "Yes":
                self.import_neg_indices.append(row[1])
            else:
                self.not_import_neg_indices.append(row[1])
            self.neg_indices.append(row[1])

        assert len(self.neg_indices) == len(data)

        def index_to_metrics(indices, setting):
            new_labels = []
            new_preds = []
            for index in indices:
                new_labels.append(labels[int(index)])
                new_preds.append(self.preds['wic']['preds'][int(index)])
            self.results_str += f'{setting}: {accuracy_score(new_labels, new_preds)}\n'
            self.results_str += f'{setting}: {classification_report(new_labels, new_preds)}\n'
        
        # Get indices of non-negations
        self.non_neg_indices = [i for i in range(len(labels)) if i not in self.neg_indices]
        index_to_metrics(self.neg_indices, 'W/ negation')
        index_to_metrics(self.non_neg_indices, 'W/O negation')
        if len(self.import_neg_indices) > 0:
            index_to_metrics(self.import_neg_indices, 'W/ negation and important')
        else:
            self.results_str += 'W/ negation and important: No data\n'
        index_to_metrics(self.not_import_neg_indices, 'W/ negation and not important')

    def qnli(self):
        labels = [1 if instance['label']=="not_entailment" else 0 for instance in self.data]
        # Get all
        self.results_str += f'All data: {accuracy_score(labels, self.preds["qnli"]["preds"])}\n'
        self.results_str += f'All data: {classification_report(labels, self.preds["qnli"]["preds"])}\n'

        # Get indices of important and not important negations
        import csv
        data = []
        # Replace 'file_path.tsv' with your TSV file's path
        with open('./annotations/important-unimportant/qnli.tsv', 'r', newline='') as file:
            reader = csv.reader(file, delimiter='\t')
            for row in reader:
                data.append(row)

        data = data[1:]
        data = data[:-1]

        for row in data:
            if row[7] == "Yes":
                self.import_neg_indices.append(row[1])
            else:
                self.not_import_neg_indices.append(row[1])
            self.neg_indices.append(row[1])

        assert len(self.neg_indices) == len(data)

        def index_to_metrics(indices, setting):
            new_labels = []
            new_preds = []
            for index in indices:
                new_labels.append(labels[int(index)])
                new_preds.append(self.preds['qnli']['preds'][int(index)])
            self.results_str += f'{setting}: {accuracy_score(new_labels, new_preds)}\n'
            self.results_str += f'{setting}: {classification_report(new_labels, new_preds)}\n'
        
        # Get indices of non-negations
        self.non_neg_indices = [i for i in range(len(labels)) if i not in self.neg_indices]
        index_to_metrics(self.neg_indices, 'W/ negation')
        index_to_metrics(self.non_neg_indices, 'W/O negation')
        if len(self.import_neg_indices) > 0:
            index_to_metrics(self.import_neg_indices, 'W/ negation and important')
        else:
            self.results_str += 'W/ negation and important: No data\n'
        index_to_metrics(self.not_import_neg_indices, 'W/ negation and not important')

    def stsb(self):
        # this one uses earson and Spearman correlations
        labels = [instance['label'] for instance in self.data]

        from scipy.stats import pearsonr, spearmanr

        # Get all
        self.results_str += f'All data pearson: {pearsonr(labels, self.preds["stsb"]["preds"])}\n'
        self.results_str += f'All data spearman: {spearmanr(labels, self.preds["stsb"]["preds"])}\n'

        # Get indices of important and not important negations
        import csv
        data = []
        # Replace 'file_path.tsv' with your TSV file's path
        with open('./annotations/important-unimportant/stsb.tsv', 'r', newline='') as file:
            reader = csv.reader(file, delimiter='\t')
            for row in reader:
                data.append(row)

        data = data[1:]

        for row in data:
            if row[7] == "Yes":
                self.import_neg_indices.append(row[1])
            else:
                self.not_import_neg_indices.append(row[1])
            self.neg_indices.append(row[1])

        assert len(self.neg_indices) == len(data)

        def index_to_metrics(indices, setting):
            new_labels = []
            new_preds = []
            for index in indices:
                new_labels.append(labels[int(index)])
                new_preds.append(self.preds['stsb']['preds'][int(index)])
            self.results_str += f'{setting} pearson: {pearsonr(new_labels, new_preds)}\n'
            self.results_str += f'{setting} spearman: {spearmanr(new_labels, new_preds)}\n'
        
        # Get indices of non-negations
        self.non_neg_indices = [i for i in range(len(labels)) if i not in self.neg_indices]
        index_to_metrics(self.neg_indices, 'W/ negation')
        index_to_metrics(self.non_neg_indices, 'W/O negation')
        if len(self.import_neg_indices) > 0:
            index_to_metrics(self.import_neg_indices, 'W/ negation and important')
        else:
            self.results_str += 'W/ negation and important: No data\n'
        index_to_metrics(self.not_import_neg_indices, 'W/ negation and not important')

if __name__=="__main__":
    exp_ids = [
        ['commonsenseqa', 'base', 'or', '1e-5'],
        ['commonsenseqa', 'large', 'or', '1e-5'],
        ['commonsenseqa', 'large', 'ch', '1e-5'],
        ['commonsenseqa', 'large', 'mo', '1e-5'],

        ['qnli', 'base', 'or', '1e-5'],
        ['qnli', 'large', 'or', '1e-5'],
        ['qnli', 'large', 'ch', '1e-5'],
        ['qnli', 'large', 'mo', '1e-5'],

        ['stsb', 'base', 'or', '1e-5'],
        ['stsb', 'large', 'or', '1e-5'],
        ['stsb', 'large', 'ch', '1e-5'],
        ['stsb', 'large', 'mo', '1e-5'],

        ['wic', 'base',  'or',  '1e-5'],
        ['wic', 'large', 'or', '1e-5'],
        ['wic', 'large', 'ch', '1e-5'],
        ['wic', 'large', 'mo', '1e-5'],

        ['wsc', 'base', 'or',  '1e-6'],
        ['wsc', 'large', 'or', '1e-6'],
        ['wsc', 'large', 'ch', '1e-6'],
        ['wsc', 'large', 'mo', '1e-6'],

        
    ]
    
    for exp_id in exp_ids:
        task, model, setting, lr = exp_id
        print(exp_id)
        m = metrics(task, model, setting, lr)
        m.read_pred_data()
        if task == 'commonsenseqa':
            m.commonsenseqa()
        elif task == 'wsc':
            m.wsc()
        elif task == 'wic':
            m.wic()
        elif task == 'qnli':
            m.qnli()
        elif task == 'stsb':
            m.stsb()
        m.write_to_text()
        


