import os

filename = "Flickr8k.token.txt"
file = os.path.join('data', 'texts', filename)

with open(file) as f:
    lines = f.readlines()


sentences = [line.split('\t')[1].strip().lower() for line in lines]

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