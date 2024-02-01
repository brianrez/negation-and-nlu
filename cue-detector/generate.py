'''
from transformers import AutoTokenizer, AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
inputs = tokenizer(["In a word, the cake is a"], return_tensors="pt")
output_ids = model.generate(inputs["input_ids"], max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0])
# Now let's take the bad words out. Please note that the tokenizer is initialized differently
tokenizer_with_prefix_space = AutoTokenizer.from_pretrained("gpt2", add_prefix_space=True)
def get_tokens_as_list(word_list):
    "Converts a sequence of words into a list of tokens"
    tokens_list = []
    for word in word_list:
        tokenized_word = tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0]
        tokens_list.append(tokenized_word)
    return tokens_list
bad_words_ids = get_tokens_as_list(word_list=["mess"])
output_ids = model.generate(
    inputs["input_ids"], max_new_tokens=5, bad_words_ids=bad_words_ids, pad_token_id=tokenizer.eos_token_id
)
print(tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0])
'''

from list import negations
# print(negations)
for i in range(len(negations)):
    negations[i] = negations[i].strip()

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = "cuda"

model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)
tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base", add_prefix_space=True)

def get_tokens_as_list(word_list):
    "Converts a sequence of words into a list of tokens"
    tokens_list = []
    for word in word_list:
        tokenized_word = tokenizer([word], add_special_tokens=False).input_ids[0]
        tokens_list.append(tokenized_word)
    return tokens_list

bad_words_ids = get_tokens_as_list(word_list=negations)

def paraphrase(
    questions,
    num_beams=5,
    num_beam_groups=5,
    num_return_sequences=1,
    repetition_penalty=10.0,
    diversity_penalty=3.0,
    no_repeat_ngram_size=2,
    temperature=0.7,
    max_length=128
):
    questions = [f"paraphrase: {question}" for question in questions]
    input_ids = tokenizer(
        questions,
        return_tensors="pt", padding="longest",
        max_length=max_length,
        truncation=True,
        # pad_token_id=tokenizer.eos_token_id
    ).input_ids.to(device)

    # print(len(input_ids))
    
    outputs = model.generate(
        input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams, num_beam_groups=num_beam_groups,
        max_length=max_length, diversity_penalty=diversity_penalty, bad_words_ids=bad_words_ids,
    )

    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return res



def runInBatch(all_sentences, batch_size=16):
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

import json
with open("./dataset/train_uniques.jsonl", 'r') as file:
    all_sentences = [json.loads(line) for line in file.readlines()]
all_sentences = runInBatch(all_sentences)
with open("./dataset/nots2/train_uniques_paraphrases.jsonl", 'w') as file:
    for item in all_sentences:
        file.write(json.dumps(item) + '\n')
print("Done with train")

with open("./dataset/val_uniques.jsonl", 'r') as file:
    all_sentences = [json.loads(line) for line in file.readlines()]

all_sentences = runInBatch(all_sentences)
with open("./dataset/nots2/val_uniques_paraphrases.jsonl", 'w') as file:
    for item in all_sentences:
        file.write(json.dumps(item) + '\n')
print("Done with val")

with open("./dataset/test_uniques.jsonl", 'r') as file:
    all_sentences = [json.loads(line) for line in file.readlines()]

all_sentences = runInBatch(all_sentences)
with open("./dataset/nots2/test_uniques_paraphrases.jsonl", 'w') as file:
    for item in all_sentences:
        file.write(json.dumps(item) + '\n')
print("Done with test")