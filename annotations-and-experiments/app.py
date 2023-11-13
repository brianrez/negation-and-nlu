import json
class jsonl:
    def read(self, path):
        with open(path) as f:
            all_sentences = [json.loads(line) for line in f.readlines()]
        return all_sentences

    def write(self, path, data):
        with open(path, "w") as f:
            for line in data:
                f.write(json.dumps(line) + "\n")



def clean_pad(path, key):
    train = jsonl().read(path + "/train.jsonl")
    val   = jsonl().read(path +   "/val.jsonl")
    # test = jsonl().read(path + "/test.jsonl")

    def clean(data, key):
        for row in data:
            row[key] = row[key].replace("Affirmative Interpretation:", "")
            row[key] = row[key].replace("<pad>", "")
            row[key] = row[key].strip()
        return data

    train = clean(train, key)
    val = clean(val, key)
    # test = clean(test, key)

    jsonl().write(path + "/train.jsonl", train)
    jsonl().write(path +   "/val.jsonl", val  )
    # jsonl().write(path + "/test.jsonl", test)

'''
clean_pad("./data/commonsenseqa/mo", "question")
clean_pad("./data/wsc/mo", "text")
clean_pad("./data/wic/mo", "sentence1")
clean_pad("./data/wic/mo", "sentence2")
clean_pad("./data/stsb/mo", "text_a")
clean_pad("./data/stsb/mo", "text_b")
'''
clean_pad("./data/qnli/mo", "premise")
clean_pad("./data/qnli/mo", "hypothesis")