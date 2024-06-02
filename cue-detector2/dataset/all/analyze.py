import json
class jsonl:
    def read(path):
        with open(path, 'r') as file:
            return [json.loads(line) for line in file.readlines()]
    def write(path, data):
        with open(path, 'w') as file:
            for item in data:
                file.write(json.dumps(item) + '\n')

train = jsonl.read("./train_uniques_paraphrases.jsonl")
val = jsonl.read("./val_uniques_paraphrases.jsonl")
test = jsonl.read("./test_uniques_paraphrases.jsonl")

# combine all the data
all_data = train + val + test
import random
random.shuffle(all_data)
# choose 100 random samples
random_samples = all_data[:100]

# add the following columns to the data
# - validated
# - status
# - notes

for item in random_samples:
    item['validated'] = False
    item['status'] = ''
    item['notes'] = ''

jsonl.write("./random_samples.jsonl", random_samples)
