from utils import Jsonl
import os
from sklearn.metrics import accuracy_score, classification_report
from statsmodels.stats.contingency_tables import mcnemar
import csv
import copy
import torch


def calc_mcnemar(y_model_true, y_baseline_true, y_model_pred, y_baseline_pred):
    # Create contingency table
    table = [[0, 0], [0, 0]]
    for i in range(len(y_model_true)):
        if (
            y_model_true[i] == y_model_pred[i]
            and y_baseline_true[i] == y_baseline_pred[i]
        ):
            table[0][0] += 1  # both methods are correct
        elif (
            y_model_true[i] != y_model_pred[i]
            and y_baseline_true[i] == y_baseline_pred[i]
        ):
            table[1][0] += 1  # only model is incorrect
        elif (
            y_model_true[i] == y_model_pred[i]
            and y_baseline_true[i] != y_baseline_pred[i]
        ):
            table[0][1] += 1  # only baseline is incorrect
        else:
            table[1][1] += 1  # both methods are incorrect

    # new_table = [[table[0][0], table[0][1]], [table[1][0], table[1][1]]]
    # table = new_table
    # Get the accuracy of each method
    # print(f"Baseline accuracy: {accuracy_score(y_model_true, y_model_pred):.2f}")
    # print(f"Model accuracy: {accuracy_score(y_baseline_true, y_baseline_pred):.2f}")

    # Run McNemar test
    result = mcnemar(table, exact=True)
    # print("Contingency table:")
    # print(table)
    # print("McNemar test result:")
    # print("Statistic: ", result.statistic)
    # print("p-value: ", result.pvalue)
    if result.pvalue < 0.05:
        print("The difference is significant")
        return True
    else:
        print("The difference is not significant")
        return False


class metrics:
    def __init__(self, task, model, setting, lr, bs=None):
        self.task = task
        self.model = model
        self.setting = setting
        self.lr = lr
        self.bs = bs

        self.data = None
        self.preds = None
        self.import_neg_indices = []
        self.not_import_neg_indices = []
        self.neg_indices = []
        self.non_neg_indices = []

        self.results_str = ""
        self.tex_results = ""

    def write_to_text(self):
        pth = f"./results/{self.task}/"
        if not os.path.exists(pth):
            os.makedirs(pth)

        with open(
            f"./results/{self.task}/{self.setting}_{self.model}_{self.lr}_{self.bs}.txt", "w"
        ) as f:
            f.write(self.results_str)

    def write_to_tex(self):
        pth = f"./results/{self.task}/"
        if not os.path.exists(pth):
            os.makedirs(pth)

        with open(
            f"./results/{self.task}/{self.setting}_{self.model}_{self.lr}_{self.bs}.tex", "w"
        ) as f:
            f.write(self.tex_results)
        print(self.tex_results)

    def read_pred_data(self):
        self.data = Jsonl().read(f"./data/{self.task}/or/val.jsonl")
        if self.bs is not None:
            self.preds = torch.load(
                f"./preds/{self.task}/{self.setting}_{self.model}_{self.lr}_{self.bs}.pt"
            )
        else:
            self.preds = torch.load(
                f"./preds/{self.task}/{self.setting}_{self.model}_{self.lr}.pt"
            )

    def commonsenseqa(self):
        pred_labels = {}
        label2id = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
        labels = [label2id[instance["answerKey"]] for instance in self.data]
        # Get all
        # self.results_str += f'All data: {accuracy_score(labels, self.preds["commonsenseqa"]["preds"])}\n'
        # self.results_str += f'All data: {classification_report(labels, self.preds["commonsenseqa"]["preds"])}\n'

        # Get indices of important and not important negations
        import csv

        data = []
        # Replace 'file_path.tsv' with your TSV file's path
        with open(
            "./annotations/important-unimportant/commonsenseqa.tsv", "r", newline=""
        ) as file:
            reader = csv.reader(file, delimiter="\t")
            for row in reader:
                data.append(row)

        data = data[1:]

        for i in range(len(self.data)):
            for j in range(len(data)):
                if self.data[i]["question"].strip() == data[j][2].strip():
                    if data[j][6] == "Yes":
                        self.import_neg_indices.append(i)
                    else:
                        self.not_import_neg_indices.append(i)
                    self.neg_indices.append(int(i))
                    break

        assert len(self.neg_indices) == len(
            data
        ), f"{len(self.neg_indices)} != {len(data)}"
        assert len(self.import_neg_indices) + len(self.not_import_neg_indices) == len(
            self.neg_indices
        )

        def index_to_metrics(indices, setting):
            new_labels = []
            new_preds = []
            for index in indices:
                new_labels.append(labels[int(index)])
                new_preds.append(self.preds["commonsenseqa"]["preds"][int(index)])
            return copy.deepcopy(new_labels), copy.deepcopy(new_preds)

        # Get indices of non-negations
        self.non_neg_indices = [
            i for i in range(len(labels)) if i not in self.neg_indices
        ]

        pred_labels["all"] = index_to_metrics([i for i in range(len(labels))], "all")
        pred_labels["non_neg"] = index_to_metrics(self.non_neg_indices, "W/O negation")
        pred_labels["neg"] = index_to_metrics(self.neg_indices, "W/ negation")
        if len(self.import_neg_indices) > 0:
            pred_labels["import_neg"] = index_to_metrics(
                self.import_neg_indices, "W/ negation and important"
            )
        else:
            pred_labels["import_neg"] = None
        pred_labels["not_import_neg"] = index_to_metrics(
            self.not_import_neg_indices, "W/ negation and not important"
        )

        return pred_labels

    def wsc(self):
        pred_labels = {}
        labels = [1 if instance["label"] is True else 0 for instance in self.data]
        # Get all
        # self.results_str += f'All data: {accuracy_score(labels, self.preds["wsc"]["preds"])}\n'
        # self.results_str += f'All data: {classification_report(labels, self.preds["wsc"]["preds"])}\n'

        # Get indices of important and not important negations
        import csv

        data = []
        # Replace 'file_path.tsv' with your TSV file's path
        with open(
            "./annotations/important-unimportant/wsc.tsv", "r", newline=""
        ) as file:
            reader = csv.reader(file, delimiter="\t")
            for row in reader:
                data.append(row)

        data = data[1:]

        for row in data:
            if row[6] == "Yes":
                self.import_neg_indices.append(int(row[1]))
            else:
                self.not_import_neg_indices.append(int(row[1]))
            self.neg_indices.append(int(row[1]))
        self.non_neg_indices = [
            i for i in range(len(labels)) if i not in self.neg_indices
        ]

        assert len(self.neg_indices) == len(data)
        assert len(self.import_neg_indices) + len(self.not_import_neg_indices) == len(
            self.neg_indices
        )

        def index_to_metrics(indices, setting):
            new_labels = []
            new_preds = []
            for index in indices:
                new_labels.append(labels[int(index)])
                new_preds.append(self.preds["wsc"]["preds"][int(index)])
            return copy.deepcopy(new_labels), copy.deepcopy(new_preds)

        # Get indices of non-negations
        self.non_neg_indices = [
            i for i in range(len(labels)) if i not in self.neg_indices
        ]

        pred_labels["all"] = index_to_metrics([i for i in range(len(labels))], "all")
        pred_labels["non_neg"] = index_to_metrics(self.non_neg_indices, "W/O negation")
        pred_labels["neg"] = index_to_metrics(self.neg_indices, "W/ negation")
        if len(self.import_neg_indices) > 0:
            pred_labels["import_neg"] = index_to_metrics(
                self.import_neg_indices, "W/ negation and important"
            )
        else:
            pred_labels["import_neg"] = None
        pred_labels["not_import_neg"] = index_to_metrics(
            self.not_import_neg_indices, "W/ negation and not important"
        )

        return pred_labels

    def wic(self):
        pred_labels = {}
        labels = [1 if instance["label"] is True else 0 for instance in self.data]

        # Get all
        # self.results_str += f'All data: {accuracy_score(labels, self.preds["wic"]["preds"])}\n'
        # self.results_str += f'All data: {classification_report(labels, self.preds["wic"]["preds"])}\n'

        # Get indices of important and not important negations
        import csv

        data = []
        # Replace 'file_path.tsv' with your TSV file's path
        with open(
            "./annotations/important-unimportant/wic.tsv", "r", newline=""
        ) as file:
            reader = csv.reader(file, delimiter="\t")
            for row in reader:
                data.append(row)

        data = data[1:]

        for row in data:
            if row[8] == "Yes":
                self.import_neg_indices.append(int(row[1]))
            else:
                self.not_import_neg_indices.append(int(row[1]))
            self.neg_indices.append(int(row[1]))

        assert len(self.neg_indices) == len(data)
        assert len(self.import_neg_indices) + len(self.not_import_neg_indices) == len(
            self.neg_indices
        )

        def index_to_metrics(indices, setting):
            new_labels = []
            new_preds = []
            for index in indices:
                new_labels.append(labels[int(index)])
                new_preds.append(self.preds["wic"]["preds"][int(index)])
            return copy.deepcopy(new_labels), copy.deepcopy(new_preds)

        # Get indices of non-negations
        self.non_neg_indices = [
            i for i in range(len(labels)) if i not in self.neg_indices
        ]

        pred_labels["all"] = index_to_metrics([i for i in range(len(labels))], "all")
        pred_labels["non_neg"] = index_to_metrics(self.non_neg_indices, "W/O negation")
        pred_labels["neg"] = index_to_metrics(self.neg_indices, "W/ negation")
        if len(self.import_neg_indices) > 0:
            pred_labels["import_neg"] = index_to_metrics(
                self.import_neg_indices, "W/ negation and important"
            )
        else:
            pred_labels["import_neg"] = None
        pred_labels["not_import_neg"] = index_to_metrics(
            self.not_import_neg_indices, "W/ negation and not important"
        )

        return pred_labels

    def qnli(self):
        pred_labels = {}
        labels = [
            1 if instance["label"] == "not_entailment" else 0 for instance in self.data
        ]
        # Get all
        # self.results_str += f'All data: {accuracy_score(labels, self.preds["qnli"]["preds"])}\n'
        # self.results_str += f'All data: {classification_report(labels, self.preds["qnli"]["preds"])}\n'

        # Get indices of important and not important negations
        import csv

        data = []
        # Replace 'file_path.tsv' with your TSV file's path
        with open(
            "./annotations/important-unimportant/qnli.tsv", "r", newline=""
        ) as file:
            reader = csv.reader(file, delimiter="\t")
            for row in reader:
                data.append(row)

        data = data[1:]
        data = data[:-1]

        for row in data:
            if row[7] == "Yes":
                self.import_neg_indices.append(int(row[1]))
            else:
                self.not_import_neg_indices.append(int(row[1]))
            self.neg_indices.append(int(row[1]))

        assert len(self.neg_indices) == len(data)
        assert len(self.import_neg_indices) + len(self.not_import_neg_indices) == len(
            self.neg_indices
        )

        def index_to_metrics(indices, setting):
            new_labels = []
            new_preds = []
            for index in indices:
                new_labels.append(labels[int(index)])
                new_preds.append(self.preds["qnli"]["preds"][int(index)])
            return copy.deepcopy(new_labels), copy.deepcopy(new_preds)

        # Get indices of non-negations
        self.non_neg_indices = [
            i for i in range(len(labels)) if i not in self.neg_indices
        ]

        pred_labels["all"] = index_to_metrics([i for i in range(len(labels))], "all")
        pred_labels["non_neg"] = index_to_metrics(self.non_neg_indices, "W/O negation")
        pred_labels["neg"] = index_to_metrics(self.neg_indices, "W/ negation")
        if len(self.import_neg_indices) > 0:
            pred_labels["import_neg"] = index_to_metrics(
                self.import_neg_indices, "W/ negation and important"
            )
        else:
            pred_labels["import_neg"] = None
        pred_labels["not_import_neg"] = index_to_metrics(
            self.not_import_neg_indices, "W/ negation and not important"
        )

        return pred_labels

    def stsb(self):
        # this one uses earson and Spearman correlations
        labels = [instance["label"] for instance in self.data]

        from scipy.stats import pearsonr, spearmanr

        # Get all
        # self.results_str += f'All data pearson: {pearsonr(labels, self.preds["stsb"]["preds"])}\n'
        # self.results_str += f'All data spearman: {spearmanr(labels, self.preds["stsb"]["preds"])}\n'

        # Get indices of important and not important negations
        import csv

        data = []
        # Replace 'file_path.tsv' with your TSV file's path
        with open(
            "./annotations/important-unimportant/stsb.tsv", "r", newline=""
        ) as file:
            reader = csv.reader(file, delimiter="\t")
            for row in reader:
                data.append(row)

        data = data[1:]

        for row in data:
            if row[7] == "Yes":
                self.import_neg_indices.append(row[1])
            else:
                self.not_import_neg_indices.append(row[1])
            self.neg_indices.append(int(row[1]))

        assert len(self.neg_indices) == len(data)
        assert len(self.import_neg_indices) + len(self.not_import_neg_indices) == len(
            self.neg_indices
        )
        print(len(self.import_neg_indices))

        def index_to_metrics(indices, setting):
            new_labels = []
            new_preds = []
            for index in indices:
                new_labels.append(labels[int(index)])
                new_preds.append(self.preds["stsb"]["preds"][int(index)])
            self.results_str += (
                f"{setting} pearson: {pearsonr(new_labels, new_preds)}\n"
            )
            self.results_str += (
                f"{setting} spearman: {spearmanr(new_labels, new_preds)}\n"
            )
            return (
                pearsonr(new_labels, new_preds).statistic,
                spearmanr(new_labels, new_preds).correlation,
            )

        # Get indices of non-negations
        self.non_neg_indices = [
            i for i in range(len(labels)) if i not in self.neg_indices
        ]
        self.tex_results += "% \all data\n"
        a, b = index_to_metrics([i for i in range(len(labels))], "all")
        self.tex_results += f" & {a:.2f} & {b:.2f} & \\\\ \n"

        self.tex_results += "% \w/o negation\n"
        a, b = index_to_metrics(self.non_neg_indices, "W/O negation")
        self.tex_results += f" & {a:.2f} & {b:.2f} & \\\\ \n"

        self.tex_results += "% \w/ negation\n"
        a, b = index_to_metrics(self.neg_indices, "W/ negation")
        self.tex_results += f" & {a:.2f} & {b:.2f} & \\\\ \n"

        if len(self.import_neg_indices) > 0:
            self.tex_results += "% \w/ negation and important\n"
            a, b = index_to_metrics(
                self.import_neg_indices, "W/ negation and important"
            )
            self.tex_results += f" & {a:.2f} & {b:.2f} & \\\\ \n"
        else:
            self.tex_results += "% \w/ negation and important: No Data\n"
            self.results_str += "W/ negation and important: No data\n"
        self.tex_results += "% \w/ negation and not important\n"
        a, b = index_to_metrics(
            self.not_import_neg_indices, "W/ negation and not important"
        )
        self.tex_results += f" & {a:.2f} & {b:.2f} & \\\\ \n"


if __name__ == "__main__":
    baseline = ['qnli', 'large', 'or', '1e-5']
    model =    ['qnli', 'large', 'mo', '1e-5', 16]

    b = metrics(baseline[0], baseline[1], baseline[2], baseline[3])
    b.read_pred_data()
    m = metrics(model[0], model[1], model[2], model[3], model[4])
    m.read_pred_data()
    if baseline[0] == "commonsenseqa":
        b_pl = b.commonsenseqa()
        m_pl = m.commonsenseqa()
    elif baseline[0] == "wsc":
        b_pl = b.wsc()
        m_pl = m.wsc()
    elif baseline[0] == "wic":
        b_pl = b.wic()
        m_pl = m.wic()
    elif baseline[0] == "qnli":
        b_pl = b.qnli()
        m_pl = m.qnli()

    for key in b_pl.keys():
        if b_pl[key] is None:
            continue
        print(key)
        calc_mcnemar(b_pl[key][0], m_pl[key][0], b_pl[key][1], m_pl[key][1])
