
'''

Script to tokenize and detokenize captions. All captions used in model
must begin with '<SOS>' and end with '<EOS>'. If a sentence is shorter
than the largest sentence in the dataset overall, then padding
are applied after '<EOS>' to make all sentences equal length that is the 
maximum length of the sentence in the dataset. 

'''

from vocabulary import inv_vocab, vocab, max_len, \
    SOS_token, EOS_token, PAD_token


def tokenizeCaptions(sentence):
    global SOS_token, EOS_token, PAD_token, vocab, max_len

    outputSentence = [vocab.get(word) for word in sentence.split()]
    outputSentence = [SOS_token] + outputSentence + [EOS_token] + \
        [PAD_token]*(max_len - len(outputSentence)-2)

    return outputSentence


def deTokenizeCaptions(tokenizedSentence, asString=False):
    global inv_vocab

    outputSentence = [inv_vocab.get(int(word)) for word in tokenizedSentence]

    if asString:
        removeTokens = ["<SOS>", "<EOS>", "<PAD>"]
        strOut = " "
        outputSentence = [
            word for word in outputSentence if word not in removeTokens
        ]
        outputSentence = strOut.join(outputSentence)

    return outputSentence


if __name__ == '__main__':
    from vocabulary import example
    token = tokenizeCaptions(example)
    deTokenList = deTokenizeCaptions(token)
    deTokenWord = deTokenizeCaptions(token, True)

    print(token)
    print(deTokenList)
    print(deTokenWord)
