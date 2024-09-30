# -*- coding: utf-8 -*-


import sys

sys.path.insert(0, "./jiant")

import argparse
import json
import os

import jiant.proj.main.export_model as export_model
import jiant.proj.main.runscript as main_runscript
import jiant.proj.main.scripts.configurator as configurator
import jiant.proj.main.tokenize_and_cache as tokenize_and_cache
import jiant.scripts.download_data.runscript as downloader
import jiant.utils.display as display
import jiant.utils.python.io as py_io
import requests


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


def run(task, model_type, setting, lr=None, batch_size=None, model_path=None):


    config_path = "./config/" + task + "/config.json"

    # Read parameters from json file
    with open(config_path) as json_file_obj:
        params = json.load(json_file_obj)
    if lr is not None:
        params["learning_rate"] = lr
    if batch_size is not None:
        params["train_batch_size"] = batch_size
        params["eval_batch_size"] = batch_size

    if params["run_config"] is not None:
        if model_type == "roberta-base":
            params["run_config"] = f"./run_configs/{task}_run_config_roberta_base.json"
        elif model_type == "roberta-large":
            params["run_config"] = f"./run_configs/{task}_run_config_roberta_large.json"

    params["model_type"] = model_type
    params["model_path"] = f"./models/{model_path}/model/model.p"
    params["model_config_path"] = f"./models/{model_type}/model/config.json"
    params["model_tokenizer_path"] = f"./models/{model_type}/tokenizer"


    os.makedirs("./run_configs/", exist_ok=True)

    # Download model
    if params["is_download_model"]:
        export_model.lookup_and_export_model(
            model_type=model_type,
            output_base_path="./models/" + model_type,
        )

    # Tokenize and cache-----------------------------------------------------
    tokenize_and_cache.main(
        tokenize_and_cache.RunConfiguration(
            task_config_path=f"./content/exp/tasks/configs/{task}_config.json",
            # model_type=MODEL_TYPE,
            # model_tokenizer_path=params["model_tokenizer_path"],
            hf_pretrained_model_name_or_path=model_type,
            output_dir=f"./outputs/{task}",
            phases=["train", "val"],
        )
    )
    print("Tokenization is completed!")

    # Write a run config -----------------------------------------------------------
    jiant_run_config = configurator.SimpleAPIMultiTaskConfigurator(
        task_config_base_path="./content/exp/tasks/configs",
        task_cache_base_path="./outputs",
        train_task_name_list=[task],
        val_task_name_list=[task],
        train_batch_size=int(params["train_batch_size"]),
        eval_batch_size=int(params["eval_batch_size"]),
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
        output_dir="./runs/" + task,
        hf_pretrained_model_name_or_path= model_path,
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
        + task
        + "/"
        + setting
        + "_"
        + model_path
        + "_"
        + str(params["learning_rate"])
        + "_"
        + str(params["train_batch_size"])
        + ".pt"
    )
    if not os.path.exists("./preds/" + task + "/"):
        os.makedirs("./preds/" + task + "/")

    os.system("cp ./runs/" + task + "/val_preds.p " + val_save_path)
    os.system("git pull")
    os.system(f"git add {val_save_path}")
    os.system(f'git commit -m " {task} {model_path} {setting} {params["learning_rate"]} {params["train_batch_size"]}"')
    os.system("git push")


# settings: "ch", "or", "mo"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default='qnli')    
    parser.add_argument('--model_name', type=str, default='roberta-large')
    parser.add_argument('--model_path', type=str, default='roberta-large')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    args = parser.parse_args()
    

    errors = open("errors.txt", "a")
    dones = open("dones.txt", "a")
    try:
        setting = "or"
        import sys

        import jiant.proj.main.export_model as export_model

        # sys.path.insert(0, "./jiant")

        try:
            export_model.export_model(
                    hf_pretrained_model_name_or_path= args.model_path,
                    output_base_path=f"./models/{args.model_path}",
                )
            os.system(f"cp -r ./models/{args.model_name}/tokenizer ./models/{args.model_path}/")
            os.system(f"cp ./models/{args.model_name}/config.json ./models/{args.model_path}/")
        except Exception as e:
            os.system(f"cp -r ./models/{args.model_name}/tokenizer ./models/{args.model_path}/")
            os.system(f"cp ./models/{args.model_name}/config.json ./models/{args.model_path}/")
            os.system(f"python3 addtokenizer.py --model_name {args.model_name} --model_path {args.model_path}")
            export_model.export_model(
                    hf_pretrained_model_name_or_path= args.model_path,
                    output_base_path=f"./models/{args.model_path}",
                )
            
        
        # task, model, setting, lr, bs = exp_id
        # move_files(task, setting)
        # run(task, model, setting, lr, bs)
        # dones.write(str(exp_id) + "\n")
        move_files(args.task_name, setting)
        run(args.task_name, args.model_name, setting, args.learning_rate, args.batch_size, args.model_path)

        # os.system("git add .")
        # os.system('git commit -m " ' + task + "_" + model + "_" + setting + ' "')
        # os.system("git push")
        notif = (
            f'''
            Training of {args.task_name} with {args.model_path} and {setting} and {args.learning_rate} is completed!"
            '''
        )
        requests.post(
            "https://ntfy.sh/mhrnlpmodels", data=notif.encode(encoding="utf-8")
        )
        errors.close()

    except Exception as e:
        notif = (
            f'''
            Training of {args.task_name} with {args.model_path} and {setting} and {args.learning_rate} failed!
            The error is: {e}
            '''
        )
        requests.post(
            "https://ntfy.sh/mhrnlpmodels",
            data=notif.encode(encoding="utf-8"),
            headers={"Priority": "5"},
        )
        raise e
        # errors.write(str(exp_id) + "\n" + str(e) + "\n")
        # errors.write("--------------------------------------------------\n")
        # errors.close()
