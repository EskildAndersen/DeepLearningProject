import os
import re

filename = "Flickr8k.token.txt"
file = os.path.join('data', 'texts', filename)

with open(file) as f:
    lines = f.readlines()

new_file = os.path.join('data', 'texts', 'labels.txt')

with open(new_file, 'w') as f:
    for line in lines:
        img, sentence = line.split('\t')

        org = sentence
        
        # All lower case letters
        sentence = sentence.lower()
        
        # Remove all 's
        isWithS = bool(re.search("'s'", sentence))
        sentence = re.sub("'s", '', sentence)

        # Replace every '-' with space
        isWithHyphen = bool(re.search('-', sentence))
        sentence = re.sub('-', ' ', sentence)

        # Remove all special characters
        isWithSpecial = bool(re.search('[@_!#$%^&*()<>?/\|}{~:,]', sentence))
        sentence = re.sub('''[@_!#$%^&*()<>?/\|}{~:.,'"]''', ' ', sentence)
        
        # Remove all double spacing
        sentence = re.sub(' +', ' ', sentence)

        output = '\t'.join([img, sentence])
        
        f.write(output)
