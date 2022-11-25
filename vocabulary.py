import os
import time

_s0 = time.time()
_filename = "labels.txt"
_file = os.path.join('data', 'texts', _filename)

with open(_file) as f:
    _lines = f.readlines()

_sentences = [line.split('\t')[1].strip().lower() for line in _lines]

max_len = max([len(s.split(' ')) for s in _sentences]) + 2
example = _sentences[0]
print(max_len)

_words = {w for s in _sentences for w in s.split(' ')}
_words |= {'<EOS>', '<SOS>', '<PAS>'}

vocab = {w : i for i, w in enumerate(_words)}
inv_vocab = {i : w for w, i in vocab.items()}

_s1 = time.time()

if __name__ == '__main__':
    print(len(_words))
    print(_s1-_s0)


