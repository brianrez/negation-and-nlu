from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = "cuda"

tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")

model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)

def paraphrase(
    questions,  # Changed to accept a list of questions
    num_beams=5,
    num_beam_groups=5,
    num_return_sequences=1,
    repetition_penalty=10.0,
    diversity_penalty=3.0,
    no_repeat_ngram_size=2,
    temperature=0.7,
    max_length=128
):
    # Tokenize a batch of sentences
    input_ids = tokenizer(
        [f'paraphrase: {question}' for question in questions],
        return_tensors="pt", padding=True,  # Modified for batch processing
        max_length=max_length, truncation=True
    ).input_ids.to(device)
    
    outputs = model.generate(
        input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams, num_beam_groups=num_beam_groups,
        max_length=max_length, diversity_penalty=diversity_penalty
    )

    # Decode each item in the batch
    res = [tokenizer.decode(outputs[i], skip_special_tokens=True) for i in range(len(questions))]

    return res


def affir_gen(sentences):  # Now accepts a list of sentences
    par1 = paraphrase(sentences)
    results = []

    for idx, sentence in enumerate(sentences):
        negated = False
        for cue in NEG_CUES:
            if cue in par1[idx]:
                negated = True
                break
        if not negated:
            results.append(par1[idx])
        else:
            par2 = paraphrase([sentence], num_return_sequences=5)[0]
            for p in par2:
                if not any(cue in p for cue in NEG_CUES):
                    results.append(p)
                    break
            else:
                results.append(par1[idx])
    
    return results


import json
import pickle

with open("./negations.pkl", "rb") as F:
    NEG_CUES = pickle.load(F)

class jsonl:
    def read(self, path):
        with open(path) as f:
            all_sentences = [json.loads(line) for line in f.readlines()]
        return all_sentences

    def write(self, path, data):
        with open(path, "w") as f:
            for line in data:
                f.write(json.dumps(line) + "\n")


def gpt_par(path, destination, key):
    train = jsonl().read(path + "/train.jsonl")
    val   = jsonl().read(path +   "/val.jsonl")
    # test = jsonl().read(path + "/test.jsonl")

    def paraphraser(data, key, batch_size=32):
        all_ = len(data)
        batches = [data[i:i + batch_size] for i in range(0, all_, batch_size)]
        i = 0
        for batch in batches:
            i += batch_size
            sentences = [row[key] for row in batch]
            paraphrased = affir_gen(sentences)

            for i, row in enumerate(batch):
                row[key] = sentences[i] + " Affirmative interpretation: " + paraphrased[i]
        
            print(f"{i}/{all_}")

        return data

    train = paraphraser(train, key)
    print("train done")
    val = paraphraser(val, key)
    print("val done")
    # test = paraphraser(test)
    # print("test done")

    jsonl().write(destination + "/train.jsonl", train)
    jsonl().write(destination +   "/val.jsonl", val  )
    # jsonl().write(destination + "/test.jsonl", test)

'''
gpt_par("./content/exp/tasks/data/commonsenseqa", "./data/commonsenseqa/ch", "question")
print("commonsenseqa done")
gpt_par("./content/exp/tasks/data/wsc", "./data/wsc/ch", "text")
print("wsc done")
gpt_par("./content/exp/tasks/data/wic", "./data/wic/ch", "sentence1")
print("wic done")
gpt_par("./data/wic/ch", "./data/wic/ch", "sentence2")
print("wic done")
gpt_par("./content/exp/tasks/data/stsb", "./data/stsb/ch", "text_a")
print("stsb done")
gpt_par("./data/stsb/ch", "./data/stsb/ch", "text_b")
print("stsb done")
'''
gpt_par("./content/exp/tasks/data/qnli", "./data/qnli/ch", "hypothesis")
print("qnli done")
gpt_par("./content/exp/tasks/data/qnli", "./data/qnli/ch", "premise")
print("qnli done")

