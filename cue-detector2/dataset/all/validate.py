from analyze import jsonl
import json

random_samples = jsonl.read("./random_samples.jsonl")
for item in random_samples:
    if not item['validated']:
        print('sentence:    ' + str(item['sentence']))
        print('paraphrase:  ' + str(item['paraphrases']))
        print()
        print("Is this a valid paraphrase? 0: same meaning, 1: different meaning, 2: wrong meaning")
        response = input()
        item['status'] = int(response)   

        item['validated'] = True

        print()
        print()
        print()
    print("Done with instance")
    # ask if the user wants to continue
    print("Continue? (y/n)")
    response = input()
    if response == 'n':
        break
jsonl.write("./random_samples.jsonl", random_samples)
        