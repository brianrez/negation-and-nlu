# -*- coding: utf-8 -*-

import torch
import numpy as np
import random
import os
import argparse
import json 
import tqdm
from transformers import (# BertTokenizer, 
                          # AlbertTokenizer, 
                          #T5Tokenizer,
                          RobertaTokenizer,
                          AutoTokenizer,
                          # BertConfig, 
                          # AdamW,
                          # get_linear_schedule_with_warmup)
                        )

from module.model import DetectNeg
from module.batch import Batchprep
from preprocessing import Dataprep, DataprepFile
import util2 as util



def set_seed(seed=42):
	os.environ['PYTHONHASHSEED'] = str(seed)
	random.seed(seed)	
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # required for using multi-GPUs.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


# Command line arguments
# python evaluate.py --config_path ./config/config.json
argParser = argparse.ArgumentParser()
argParser.add_argument("-c", "--config_path", help="path of the configuration file", required=True) 
argParser.add_argument("--input_path", help="path to the input file", required=False, default=None) 
argParser.add_argument("--output_path", help="path to the output file", required=False, default=None)      
args        = argParser.parse_args()
config_path = args.config_path
input_path  = args.input_path
output_path = args.output_path



# Read parameters from json file
with open(config_path) as json_file_obj: 
	params = json.load(json_file_obj)


# Set the seed    
set_seed(params["seed"]) 

model_path = params["best_model_path"]
eval_paths = [ (name, path) for name, path in params["evaluate_paths"].items()]
    
# Model Load
map_location = 'cuda:{}'.format(params["device"]) if params["use_gpu"] else 'cpu'
state = torch.load(model_path, map_location=map_location)

vocabs = state['vocabs']
model = DetectNeg(params, vocabs)
model.load_state_dict(state['model'])

if torch.cuda.is_available()==True: 
    device = torch.device("cuda:"+str(params["device"]))
else: 
    device = torch.device("cpu") 

model.to(device) 
model.eval()  

# Get the tokenizer
tokenizer = RobertaTokenizer.from_pretrained(params["RoBERTa-base-path"], do_lower_case=False)
# tokenizer = tokenizer(

def negCues(sents):
    '''
        - negated: an array of boolean values, where True indicates that the sentence is negated.
        - cues: an array of strings, where each string is a cue word that indicates negation.
    '''
    dev_data = DataprepFile().preprocess(sents)
    dev_size = len(dev_data["tokens"])

    dev_output = []
    dev_batch_num = int(np.ceil(dev_size/params["dev_batch_size"]))
    dev_iterator = Batchprep().get_a_batch(params, vocabs, tokenizer, dev_data, dev_size, dev_batch_num, device, shuffle=False)
    for dev_batch_idx in range(dev_batch_num):
        dev_batch_data, dev_batch_labels = next(dev_iterator)
        dev_batch_output = model.predict(dev_batch_data)
        dev_output.extend(dev_batch_output)   
    negated, cues = util.predict_only(dev_output, dev_data, vocabs)

    return negated, cues


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = "cuda"

model2 = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)
tokenizer2 = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base", add_prefix_space=True)

def get_tokens_as_list(word_list):
    "Converts a sequence of words into a list of tokens"
    tokens_list = []
    for word in word_list:
        tokenized_word = tokenizer2([word], add_special_tokens=False).input_ids[0]
        tokens_list.append(tokenized_word)
    return tokens_list

# bad_words_ids = get_tokens_as_list(word_list=negations)

def paraphrase(
    questions,
    num_beams=5,
    num_beam_groups=5,
    num_return_sequences=1,
    repetition_penalty=10.0,
    diversity_penalty=3.0,
    no_repeat_ngram_size=2,
    temperature=0.7,
    max_length=128,
    bad_words_ids=None
):
    questions = [f"paraphrase: {question}" for question in questions]
    input_ids = tokenizer2(
        questions,
        return_tensors="pt", padding="longest",
        max_length=max_length,
        truncation=True,
        # pad_token_id=tokenizer.eos_token_id
    ).input_ids.to(device)

    # print(len(input_ids))
    
    outputs = model2.generate(
        input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams, num_beam_groups=num_beam_groups,
        max_length=max_length, diversity_penalty=diversity_penalty, bad_words_ids=bad_words_ids,
    )

    res = tokenizer2.batch_decode(outputs, skip_special_tokens=True)

    return res


import pickle
import os
def runInBatch(all_sentences, batch_size=8):
    # create a dict
    pared = 0
    modelCalled = 0
    tracker = {}
    for i in range(len(all_sentences)):
        all_sentences[i]['paraphrased'] = False
        tracker[i] = all_sentences[i]
    
    def createBatch():
        batch = {}
        found = 0

        for key in tracker:
            if tracker[key]['paraphrased'] == False:
                batch[key] = tracker[key]['sentence']
                found += 1
                if found == batch_size:
                    break
        return batch
    
    def hasUnparaphrased():
        for key in tracker:
            if tracker[key]['paraphrased'] == False:
                return True
        return False
    
    negations = ["not", "no", "never"]
    bad_words_ids = get_tokens_as_list(word_list=negations)
    
    while hasUnparaphrased():
        batch = createBatch()
        # print(f"Batch size: {len(batch)}")
        batchItems = [batch[item]['sentence'] for item in batch]
        bad_words_ids = get_tokens_as_list(word_list=negations)
        paraphrases = paraphrase(batchItems, bad_words_ids=bad_words_ids)
        modelCalled += 1
        negated, cues = negCues(paraphrases)

        for cue in cues: 
            if cue.strip() not in negations:
                negations.append(cue.strip())
        
        m = 0
        for key in batch:
            if not negated[m]:
                tracker[key]['paraphrased'] = True
                tracker[key]['paraphrases'] = paraphrases[m]
                pared += 1
            m += 1
        print(f"Done with {pared} out of {len(all_sentences)} instances. current size of negations: {len(negations)}, model called: {modelCalled}", end='\r')
    
    
    try:
        # check if negation.pkl exists
        if os.path.exists("negation.pkl"):
            with open("negation.pkl", "rb") as file:
                negations2 = pickle.load(file)
            negations2 = list(set(negations2 + negations))
        else:
            negations2 = negations
        with open("negation.pkl", "wb") as file:
            pickle.dump(negations2, file)
    except Exception as e:
        print(e)

    return [tracker[key] for key in tracker]

    '''
    new_sentences = []
    for i in range(0, len(all_sentences), batch_size):
        thisBatch = all_sentences[i:i+batch_size]
        sentences = [item['sentence'] for item in thisBatch]
        paraphrases = paraphrase(sentences)
        # print(len(sentences))
        # print(paraphrases)
        for j, item in enumerate(thisBatch):
            item['paraphrases'] = paraphrases[j]
            new_sentences.append(item)
        print(f"Done with instances {i} to {i+batch_size} out of {len(all_sentences)} instances.", end='\r')
    return new_sentences
    '''


import json
with open("./dataset/train_uniques.jsonl", 'r') as file:
    all_sentences = [json.loads(line) for line in file.readlines()]
all_sentences = runInBatch(all_sentences)
with open("./dataset/all/train_uniques_paraphrases.jsonl", 'w') as file:
    for item in all_sentences:
        file.write(json.dumps(item) + '\n')
print("Done with train")

with open("./dataset/val_uniques.jsonl", 'r') as file:
    all_sentences = [json.loads(line) for line in file.readlines()]

all_sentences = runInBatch(all_sentences)
with open("./dataset/all/val_uniques_paraphrases.jsonl", 'w') as file:
    for item in all_sentences:
        file.write(json.dumps(item) + '\n')
print("Done with val")

with open("./dataset/test_uniques.jsonl", 'r') as file:
    all_sentences = [json.loads(line) for line in file.readlines()]

all_sentences = runInBatch(all_sentences)
with open("./dataset/all/test_uniques_paraphrases.jsonl", 'w') as file:
    for item in all_sentences:
        file.write(json.dumps(item) + '\n')
print("Done with test")