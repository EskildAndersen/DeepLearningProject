import os
import time

_s0 = time.time()
_filename = "labels.txt"
_file = os.path.join('data', 'texts', _filename)

with open(_file) as f:
    _lines = f.readlines()

_sentences = [line.split('\t')[1].strip().lower() for line in _lines]

_words = {w for s in _sentences for w in s.split(' ')}
print(len(_words))

vocab = {w : i for i, w in enumerate(_words)}

inv_vocab = {i : w for w, i in vocab.items()}

_s1 = time.time()

print(_s1-_s0)


