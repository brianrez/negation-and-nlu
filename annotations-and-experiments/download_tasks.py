# -*- coding: utf-8 -*-

import jiant.proj.main.export_model as export_model
import sys
sys.path.insert(0, "./jiant")


import os

#import jiant.utils.python.io as py_io
import jiant.scripts.download_data.runscript as downloader

# DATA_DIR = "./content/exp/tasks"

# # Data -------------------------------------------
# os.makedirs(DATA_DIR, exist_ok=True)

# task_list = ["stsb", "qnli", "wsc", "wic", "commonsenseqa"]
# # task_list = []

# for task in task_list:
#     print("task: {}\n".format(task))
#     downloader.download_data([task], DATA_DIR)



export_model.export_model(
        hf_pretrained_model_name_or_path="roberta-large",
        output_base_path="./models/roberta-large",
    )
