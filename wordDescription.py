import os
import numpy as np

filename = "Flickr8k.token.txt"
file = os.path.join('data', 'texts', filename)

with open(file) as f:
    lines = f.readlines()


sentences = [line.split('\t')[1].strip().lower() for line in lines]
sentences_len = [len(s.split(' ')[:-1]) for s in sentences]
sentences_len_sorted = sorted(sentences_len, reverse=True)

i = np.argmax(sentences_len)
print(i)

pass

words = {}

for s in sentences:
    sWords = s.split(' ')[:-1]
    
    for w in sWords:
        words[w] = words.get(w, 0) + 1

words = {k : v for k, v in sorted(words.items(), 
                                  key = lambda item: item[1], 
                                  reverse = True)}


print(words)
print(len(words))

w = '-'
print(f'{w.__repr__()} : {words.get(w)}')