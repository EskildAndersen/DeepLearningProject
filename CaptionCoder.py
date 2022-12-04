# script der skal start token,end token og pad sætningen til en maks længde
# input er En sætning, et vocab og en maks længde1

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

    outputSentence = [inv_vocab.get(word) for word in tokenizedSentence]

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
