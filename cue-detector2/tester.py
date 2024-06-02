import pickle

with open('negation.pkl', 'rb') as file:
    negations = pickle.load(file)

# save it as a txt file
with open('negation.txt', 'w') as file:
    for item in negations:
        file.write(item + '\n')