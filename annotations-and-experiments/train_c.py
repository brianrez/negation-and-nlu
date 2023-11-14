# -*- coding: utf-8 -*-


import sys

sys.path.insert(0, "./jiant")

import os

import jiant.utils.python.io as py_io
import jiant.scripts.download_data.runscript as downloader

import jiant.proj.main.tokenize_and_cache as tokenize_and_cache
import jiant.proj.main.export_model as export_model
import jiant.proj.main.scripts.configurator as configurator
import jiant.proj.main.runscript as main_runscript
import jiant.utils.display as display

import argparse
import json
import requests

"""
argParser = argparse.ArgumentParser()
argParser.add_argument("-c", "--config_path", help="path of the configuration file", required=True)      
args        = argParser.parse_args()
config_path = args.config_path
"""


def move_files(task, setting):
    os.system("rm -rf ./content/exp/tasks/data/" + task + "/train.jsonl")
    os.system("rm -rf ./content/exp/tasks/data/" + task + "/val.jsonl")

    os.system(
        "cp ./data/"
        + task
        + "/"
        + setting
        + "/train.jsonl ./content/exp/tasks/data/"
        + task
        + "/train.jsonl"
    )
    os.system(
        "cp ./data/"
        + task
        + "/"
        + setting
        + "/val.jsonl ./content/exp/tasks/data/"
        + task
        + "/val.jsonl"
    )


def run(task, model, setting, lr=None, batch_size=None):
    TASK_NAME = task
    MODEL_TYPE = model

    config_path = "./config/" + TASK_NAME + "/config.json"

    # Read parameters from json file
    with open(config_path) as json_file_obj:
        params = json.load(json_file_obj)
    if lr is not None:
        params["learning_rate"] = lr
    if batch_size is not None:
        params["train_batch_size"] = batch_size
        params["eval_batch_size"] = batch_size

    for key in params:
        if MODEL_TYPE == "roberta-base":
            if type(params[key]) == str:
                params[key] = params[key].replace("larege", "base")
        elif MODEL_TYPE == "roberta-large":
            if type(params[key]) == str:
                params[key] = params[key].replace("base", "large")

    os.makedirs("./run_configs/", exist_ok=True)

    # Download model
    if params["is_download_model"]:
        export_model.lookup_and_export_model(
            model_type=MODEL_TYPE,
            output_base_path="./models/" + MODEL_TYPE,
        )

    # Tokenize and cache-----------------------------------------------------
    tokenize_and_cache.main(
        tokenize_and_cache.RunConfiguration(
            task_config_path=f"./content/exp/tasks/configs/{TASK_NAME}_config.json",
            # model_type=MODEL_TYPE,
            # model_tokenizer_path=params["model_tokenizer_path"],
            hf_pretrained_model_name_or_path=MODEL_TYPE,
            output_dir=f"./outputs/{TASK_NAME}",
            phases=["train", "val"],
        )
    )
    print("Tokenization is completed!")

    # Write a run config -----------------------------------------------------------
    jiant_run_config = configurator.SimpleAPIMultiTaskConfigurator(
        task_config_base_path="./content/exp/tasks/configs",
        task_cache_base_path="./outputs",
        train_task_name_list=[TASK_NAME],
        val_task_name_list=[TASK_NAME],
        train_batch_size=params["train_batch_size"],
        eval_batch_size=params["eval_batch_size"],
        epochs=params["num_epochs"],
        num_gpus=params["num_gpus"],
        # warmup_steps_proportion=params["warmup_steps_proportion"]
    ).create_config()
    py_io.write_json(jiant_run_config, params["run_config"])
    display.show_json(jiant_run_config)
    print("Configuration is set up!")

    # Start training ------------------------------------------------------------------------
    print("Training Started----------")
    run_args = main_runscript.RunConfiguration(
        jiant_task_container_config_path=params["run_config"],
        output_dir="./runs/" + TASK_NAME,
        hf_pretrained_model_name_or_path=MODEL_TYPE,
        # model_type=MODEL_TYPE,
        model_path=params["model_path"],
        model_config_path=params["model_config_path"],
        # model_tokenizer_path=params["model_tokenizer_path"],
        learning_rate=float(params["learning_rate"]),
        eval_every_steps=50000,
        do_train=True,
        do_val=True,
        do_save=True,
        force_overwrite=True,
        write_val_preds=True,
        seed=params["seed"],
    )
    main_runscript.run_loop(run_args)
    print("Training is completed!")

    val_save_path = (
        "./preds/"
        + TASK_NAME
        + "/"
        + setting
        + "_"
        + MODEL_TYPE[8:]
        + "_"
        + str(params["learning_rate"])
        + "_"
        + str(params["train_batch_size"])
        + ".pt"
    )
    if not os.path.exists("./preds/" + TASK_NAME + "/"):
        os.makedirs("./preds/" + TASK_NAME + "/")

    os.system("cp ./runs/" + TASK_NAME + "/val_preds.p " + val_save_path)


# settings: "ch", "or", "mo"
if __name__ == "__main__":
    exp_ids = [
        ["commonsenseqa", "roberta-large", "ch", "1e-6"],
        ["commonsenseqa", "roberta-large", "ch", "5e-6"],
        ["commonsenseqa", "roberta-large", "ch", "5e-5"],
        ["commonsenseqa", "roberta-large", "ch", "1e-4"],
    ]

    for exp_id in exp_ids:
        errors = open("errors.txt", "a")
        dones = open("dones.txt", "a")
        try:
            task, model, setting, lr = exp_id
            move_files(task, setting)
            run(task, model, setting ,lr)
            dones.write(str(exp_id) + "\n")

            os.system("git add .")
            os.system('git commit -m " ' + task + "_" + model + "_" + setting + ' "')
            os.system("git push")
            notif = (
                "Training of "
                + task
                + " with "
                + model
                + " and "
                + setting 
                + " and "
                + lr
                + " is done!"
            )
            requests.post(
                "https://ntfy.sh/mhrnlpmodels", data=notif.encode(encoding="utf-8")
            )
            errors.close()
        except Exception as e:
            notif = (
                "Training of "
                + task
                + " with "
                + model
                + " and "
                + setting
                + " and "
                + lr
                + " is failed!"
            )
            requests.post(
                "https://ntfy.sh/mhrnlpmodels",
                data=notif.encode(encoding="utf-8"),
                headers={"Priority": "5"},
            )
            errors.write(str(exp_id) + "\n" + str(e) + "\n")
            errors.write("--------------------------------------------------\n")
            errors.close()
